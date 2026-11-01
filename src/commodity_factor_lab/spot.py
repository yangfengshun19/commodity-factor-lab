"""Physical spot quote parsing and descriptive matching; no prediction or execution."""

import math
import re
from html.parser import HTMLParser

import numpy as np
import pandas as pd

INSTRUMENTS = {
    "Au99.95": ("au", "physical_cash_t0", "CNY/g"),
    "Au99.99": ("au", "physical_cash_t0", "CNY/g"),
    "Ag99.99": ("ag", "physical_spot_t2", "CNY/kg"),
}
REJECTED = {"Au(T+D)", "Ag(T+D)", "mAu(T+D)", "iAu99.99"}


class QuoteHTML(HTMLParser):
    """Read table text and article headings; comments/scripts are never executed."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self.headings = [], []
        self.table, self.row, self.cell, self.heading = None, None, None, None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            if self.table is not None:
                raise ValueError("Nested quote tables are not supported")
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []
        elif tag == "h1":
            self.heading = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)
        if self.heading is not None:
            self.heading.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None
        elif tag == "h1" and self.heading is not None:
            self.headings.append("".join(self.heading).strip())
            self.heading = None


def _number(text: str) -> float:
    value = text.strip()
    if value in {"", "-", "--", "—", "–"}:
        return np.nan
    if not re.fullmatch(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", value):
        raise ValueError(f"Unexpected quote number: {text!r}")
    return float(value.replace(",", ""))


def parse_spot_page(content: bytes, expected_date: str, mode: str) -> tuple[pd.DataFrame, dict]:
    """Require date identity from the table or article, never from the URL alone."""
    date = pd.Timestamp(expected_date)
    if date.strftime("%Y-%m-%d") != expected_date or mode not in {
        "historical_article",
        "dated_query",
    }:
        raise ValueError("A strict ISO date and an explicit source mode are required")
    html = content.decode("utf-8-sig", errors="strict")
    parser = QuoteHTML()
    parser.feed(html)
    matches = []
    for tab in parser.tables:
        if not tab:
            continue
        header = [re.sub(r"\s+", "", cell) for cell in tab[0]]
        if "合约" in header and "收盘价" in header:
            matches.append((header, tab[1:]))
    if len(matches) != 1:
        raise ValueError("Expected exactly one identifiable daily quote table")
    header, source_rows = matches[0]
    if mode == "historical_article":
        headings = re.findall(
            r"上海黄金交易所(\d{4})年(\d{1,2})月(\d{1,2})日交易行情", " ".join(parser.headings)
        )
        if len(headings) != 1 or tuple(map(int, headings[0])) != (date.year, date.month, date.day):
            raise ValueError("Historical article heading does not identify the requested date")
        if "日期" in header:
            raise ValueError("Unexpected date layout in historical article")
    elif "日期" not in header:
        raise ValueError("Dated query requires explicit row dates")
    names = {
        "symbol": "合约",
        "open": "开盘价",
        "high": "最高价",
        "low": "最低价",
        "close": "收盘价",
        "volume_as_published": "成交量",
        "turnover_as_published": "成交金额",
    }
    indexes = {}
    for name, label in names.items():
        indexes_found = [
            i for i, text in enumerate(header) if text == label or text.startswith(label + "（")
        ]
        if len(indexes_found) != 1:
            raise ValueError(f"Missing or ambiguous quote column: {label}")
        indexes[name] = indexes_found[0]
    parsed, seen = {}, set()
    for cells in source_rows:
        if not any(cells):
            continue
        if len(cells) != len(header):
            raise ValueError("Quote row width differs from its header")
        if mode == "dated_query" and cells[header.index("日期")] != expected_date:
            raise ValueError("Query returned another date; latest/nearby fallback is forbidden")
        symbol = cells[indexes["symbol"]]
        if not symbol or symbol in seen:
            raise ValueError("Empty or duplicate daily instrument")
        seen.add(symbol)
        if symbol in INSTRUMENTS:
            parsed[symbol] = {
                name: _number(cells[i]) for name, i in indexes.items() if name != "symbol"
            }
    rows = []
    for symbol, (code, kind, unit) in INSTRUMENTS.items():
        item = {
            "date": date,
            "symbol": symbol,
            "code": code,
            "kind": kind,
            "quote_unit": unit,
            **{key: np.nan for key in names if key != "symbol"},
        }
        status = "symbol_absent" if source_rows else "empty_daily_table"
        if symbol in parsed:
            item.update(parsed[symbol])
            o, h, low, c = (item[key] for key in ("open", "high", "low", "close"))
            volume, turnover = item["volume_as_published"], item["turnover_as_published"]
            if not math.isfinite(volume) or volume < 0:
                status = "invalid_volume"
            elif volume == 0:
                status = "no_trade"
            elif not all(math.isfinite(p) and p > 0 for p in (o, h, low, c)):
                status = "missing_or_invalid_ohlc"
            elif low > min(o, c) or h < max(o, c) or low > h:
                status = "invalid_ohlc_range"
            elif not math.isfinite(turnover) or turnover <= 0:
                status = "invalid_turnover"
            else:
                status = "quote_ok"
        item.update(
            quote_status=status, publisher_available_at=pd.NaT, first_release_certified=False
        )
        rows.append(item)
    audit = {
        "date": expected_date,
        "mode": mode,
        "table_rows": len(source_rows),
        "observed_instruments": len(seen),
        "rejected_control_instruments_present": len(seen & REJECTED),
        "date_identity": "article_heading"
        if mode == "historical_article"
        else "explicit_table_rows"
        if source_rows
        else "empty_table_no_date_claim",
        "publisher_clock": "not_certified",
        "volume_counting": "two_sided" if "成交量和成交金额为双向计量" in html else "unknown",
    }
    return pd.DataFrame(rows), audit


def match_spot_diagnostics(quotes: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    """Same-date frozen selected contract only; gold quote types are one underlying."""
    if quotes.duplicated(["date", "symbol"]).any() or curves.duplicated(["date", "code"]).any():
        raise ValueError("Quote/selected-contract identities must be unique")
    for row in quotes.itertuples():
        if (
            row.symbol not in INSTRUMENTS
            or (row.code, row.kind, row.quote_unit) != INSTRUMENTS[row.symbol]
        ):
            raise ValueError(
                "Only explicitly mapped physical instruments and quote units are allowed"
            )
    selected = curves[["date", "code", "front_contract", "reference_settle", "status"]].rename(
        columns={"status": "curve_status"}
    )
    result = quotes.merge(selected, on=["date", "code"], how="left", validate="many_to_one")
    for row in result.dropna(subset=["front_contract"]).itertuples():
        if not re.fullmatch(re.escape(row.code) + r"\d{4}", row.front_contract):
            raise ValueError(
                "Selected futures identity is not a dated contract of the mapped metal"
            )
    reference = result.reference_settle
    valid = (
        result.quote_status.eq("quote_ok")
        & result.curve_status.eq("ok")
        & np.isfinite(reference)
        & reference.gt(0)
    )
    result["quote_multiplier"] = 1.0  # Both sides already use CNY/g or CNY/kg, not lot values.
    result["diagnostic_log_quote_ratio"] = np.nan
    result.loc[valid, "diagnostic_log_quote_ratio"] = np.log(
        result.loc[valid, "close"] / reference[valid]
    )
    result["diagnostic_available"] = valid
    result["publisher_clock_verified"] = False
    result["brand_location_tax_history_verified"] = False
    result["simultaneous_mark_verified"] = False
    result["strict_spot_basis"] = np.nan
    result["claim_eligible"] = False
    result["gold_withdrawal_policy_regime"] = np.where(
        result.code.eq("au"),
        np.where(
            result.date.ge("2025-11-01"), "post_2025_11_01_unmapped", "pre_2025_11_01_unmapped"
        ),
        "not_gold",
    )
    return result
