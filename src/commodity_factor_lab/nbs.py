"""NBS period reference prices: explicit publication clocks, identities and event alignment."""

import calendar
import hashlib
import math
import re
import unicodedata
from html import unescape

import pandas as pd

from .spot import QuoteHTML

PERIODS = {"上": "early", "中": "middle", "下": "late"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"注\d+", "", text)
    return text.replace("—", "-").replace("–", "-").replace("Φ", "φ")


def _plain(html: str) -> str:
    html = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    html = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", html, flags=re.S | re.I)
    return unescape(re.sub(r"<[^>]+>", " ", html))


def period_dates(period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    match = re.fullmatch(r"(20\d{2})-(0[1-9]|1[0-2])-(early|middle|late)", period)
    if match is None:
        raise ValueError("NBS period must be YYYY-MM-early/middle/late")
    year, month = map(int, match.groups()[:2])
    part = match[3]
    start = {"early": 1, "middle": 11, "late": 21}[part]
    end = {"early": 10, "middle": 20, "late": calendar.monthrange(year, month)[1]}[part]
    return pd.Timestamp(year=year, month=month, day=start), pd.Timestamp(
        year=year, month=month, day=end
    )


def _number(text: str) -> float:
    value = normalize(text)
    if value in {"", "-", "--"}:
        return float("nan")
    if not re.fullmatch(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", value):
        raise ValueError(f"Unexpected NBS numeric cell: {text!r}")
    result = float(value.replace(",", ""))
    if not math.isfinite(result):
        raise ValueError("NBS numeric cell must be finite")
    return result


def _unique_table(tables: list, first_header: str, width: int) -> tuple[list, int]:
    matches = []
    for tab in tables:
        if tab and tab[0] and normalize(tab[0][0]) == first_header and len(tab[0]) == width:
            matches.append([[normalize(cell) for cell in row] for row in tab])
    if not matches or any(tab != matches[0] for tab in matches[1:]):
        raise ValueError("NBS quote/specification table missing or desktop/mobile copies conflict")
    return matches[0], len(matches)


def parse_nbs_release(
    content: bytes, expected_period: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Parse the displayed clock, never URL dates, creation metadata or HTTP modification."""
    html = content.decode("utf-8-sig", errors="strict")
    titles = re.findall(r"<title\b[^>]*>(.*?)</title>", html, re.S | re.I)
    if len(titles) != 1:
        raise ValueError("Expected a single NBS article title")
    title = normalize(_plain(titles[0]))
    match = re.search(
        r"(20\d{2})年(\d{1,2})月([上中下])旬流通领域重要生产资料市场价格变动情况", title
    )
    if match is None:
        raise ValueError("Article is not the requested NBS price release")
    period = f"{match[1]}-{int(match[2]):02d}-{PERIODS[match[3]]}"
    if period != expected_period:
        raise ValueError("NBS article period differs from the requested source identity")
    start, end = period_dates(period)
    before_table = html[: html.lower().find("<table")]
    clock_text = _plain(before_table)
    clocks = set(re.findall(r"20\d{2}[/-]\d{2}[/-]\d{2}\s+\d{2}:\d{2}(?::\d{2})?", clock_text))
    if len(clocks) != 1:
        raise ValueError("NBS header requires one explicit, unambiguous publication datetime")
    available = (
        pd.Timestamp(next(iter(clocks)).replace("/", "-"))
        .tz_localize("Asia/Shanghai")
        .tz_convert("UTC")
    )
    if available.tz_convert("Asia/Shanghai").tz_localize(None).normalize() <= end:
        raise ValueError("Period reference price cannot be published before its period is complete")
    clean_text = normalize(_plain(html))
    if "据对24个省" in clean_text:
        scope = "24_provinces"
    elif "监测范围涵盖全国31个省" in clean_text:
        scope = "nationwide_31_provinces"
    else:
        scope = "scope_not_verified"
    parser = QuoteHTML()
    parser.feed(html)
    prices, price_copies = _unique_table(parser.tables, "产品名称", 5)
    specs, spec_copies = _unique_table(parser.tables, "序号", 4)
    if prices[0][1] != "单位" or re.fullmatch(r"本期价格\d*\(元\)\d*", prices[0][2]) is None:
        raise ValueError("Unexpected NBS price unit/value columns")
    specification_rows, spec_lookup = [], {}
    for row in specs[1:]:
        if not row or not row[0].isdigit():
            continue
        if len(row) != 4:
            raise ValueError("Unexpected NBS specification row width")
        key = normalize(row[1] + "(" + row[2] + ")")
        if key in spec_lookup:
            raise ValueError("Duplicate NBS product specification")
        spec_lookup[key] = row[3]
        specification_rows.append(
            {
                "period": period,
                "number": int(row[0]),
                "product": row[1],
                "specification": row[2],
                "description": row[3],
            }
        )
    rows, seen = [], set()
    for row in prices[1:]:
        if (
            not row
            or not row[0]
            or re.match(r"[一二三四五六七八九]、", row[0])
            or row[0].startswith("注")
        ):
            continue
        if len(row) != 5:
            raise ValueError("Unexpected NBS price row width")
        product = row[0]
        if product in seen:
            raise ValueError("Duplicate product inside a source price table")
        seen.add(product)
        price = _number(row[2])
        if not math.isnan(price) and price <= 0:
            raise ValueError("NBS reference price must be positive or explicitly missing")
        if row[1] not in {"吨", "千克"}:
            raise ValueError("Unsupported NBS published quantity unit")
        specification = spec_lookup.get(product, "")
        series_payload = "|".join((product, row[1], scope, specification))
        rows.append(
            {
                "period": period,
                "period_start": start,
                "period_end": end,
                "product": product,
                "quantity_unit": row[1],
                "quote_unit": "CNY/" + row[1],
                "reference_price": price,
                "reported_change": _number(row[3]),
                "source_declared_published_at": available,
                "monitoring_scope": scope,
                "specification_description": specification,
                "comparability_key": hashlib.sha256(series_payload.encode()).hexdigest(),
                "source_kind": "period_national_distribution_reference",
                "tax_basis": "publisher_includes_distribution_costs_profit_and_taxes",
                "historical_first_vintage_certified": False,
                "executable_quote": False,
            }
        )
    if len(rows) != 50 or len(specification_rows) != 50:
        raise ValueError("Fixed assay expects exactly 50 price and specification products")
    audit = {
        "period": period,
        "period_end": end,
        "source_declared_published_at": available,
        "products": len(rows),
        "specifications": len(specification_rows),
        "identical_price_table_copies": price_copies,
        "identical_spec_table_copies": spec_copies,
        "monitoring_scope": scope,
        "publication_clock_basis": "displayed_article_header_not_url_or_http_mtime",
    }
    return pd.DataFrame(rows), pd.DataFrame(specification_rows), audit


REFERENCE_MAPPING = {
    normalize(label): (code, status)
    for label, code, status in [
        ("电解铜（1#）", "cu", "reference_only"),
        ("铝锭（A00）", "al", "reference_only"),
        ("锌锭（0#）", "zn", "reference_only"),
        ("铅锭（1#）", "pb", "reference_only"),
        ("螺纹钢（Φ20mm，HRB400E）", "rb", "reference_only"),
        ("螺纹钢（Φ16-25mm，HRB400E）", "rb", "specification_break"),
        ("热轧普通板卷（4.75-11.5mm，Q235）", "hc", "reference_only"),
        ("热轧普通薄板（3mm，Q235）", "hc", "different_product_form"),
        ("天然橡胶（标准胶SCRWF）", "ru", "reference_only"),
        ("纸浆（进口针叶浆）", "sp", "reference_only"),
        ("纸浆（漂白化学浆）", "sp", "ambiguous_pulp_grade"),
    ]
}


def map_industrial_references(prices: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    if prices.duplicated(["period", "product"]).any():
        raise ValueError("Reference prices must retain one product per published period")
    result = prices.copy()
    mapped = [
        REFERENCE_MAPPING.get(normalize(label), ("", "outside_registered_mapping"))
        for label in result["product"]
    ]
    result["related_code"] = [item[0] for item in mapped]
    result["mapping_status"] = [item[1] for item in mapped]
    result["in_existing_universe"] = result.related_code.isin(universe)
    result["reference_mapping_usable"] = (
        result.in_existing_universe
        & result.mapping_status.eq("reference_only")
        & result.quantity_unit.eq("吨")
        & result.reference_price.notna()
        & result.specification_description.ne("")
        & result.monitoring_scope.isin({"24_provinces", "nationwide_31_provinces"})
    )
    result["strict_spot_basis_eligible"] = False
    return result


def align_release_events(references: pd.DataFrame, calendar_frame: pd.DataFrame) -> pd.DataFrame:
    """First observed futures signal cutoff >= declared publication. No daily backfill."""
    calendar_frame = calendar_frame.sort_values("date").copy()
    if calendar_frame.date.duplicated().any():
        raise ValueError("Signal calendar dates must be unique")
    cutoffs = pd.DatetimeIndex(pd.to_datetime(calendar_frame.signal_cutoff, utc=True))
    if cutoffs.hasnans or not cutoffs.is_monotonic_increasing or not cutoffs.is_unique:
        raise ValueError("Signal cutoffs must be known, strictly increasing instants")
    events = references[["period", "period_end", "source_declared_published_at"]].drop_duplicates()
    if events.period.duplicated().any():
        raise ValueError("Conflicting publication versions need an explicit vintage protocol")
    rows = []
    for event in events.itertuples(index=False):
        published = pd.Timestamp(event.source_declared_published_at)
        if published.tzinfo is None:
            raise ValueError("Publication timezone must be explicit")
        published = published.tz_convert("UTC")
        index = cutoffs.searchsorted(published, side="left")
        item = {
            "period": event.period,
            "period_end": event.period_end,
            "source_declared_published_at": published,
            "first_signal_date": pd.NaT,
            "first_signal_cutoff": pd.NaT,
            "previous_signal_date": pd.NaT,
            "previous_signal_cutoff": pd.NaT,
            "status": "outside_frozen_calendar",
            "claim_eligible": False,
        }
        if index < len(calendar_frame):
            item.update(
                first_signal_date=calendar_frame.iloc[index].date,
                first_signal_cutoff=cutoffs[index],
                status="source_clock_event_alignment_only",
            )
        if index:
            item.update(
                previous_signal_date=calendar_frame.iloc[
                    min(index - 1, len(calendar_frame) - 1)
                ].date,
                previous_signal_cutoff=cutoffs[min(index - 1, len(calendar_frame) - 1)],
            )
        rows.append(item)
    return pd.DataFrame(rows)
