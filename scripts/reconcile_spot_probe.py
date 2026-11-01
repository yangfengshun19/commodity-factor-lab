"""Independent raw-HTML arithmetic checks; imports no production spot parser or factor."""

import argparse
import csv
import hashlib
import json
import math
import re
from html import escape, unescape
from pathlib import Path


def reconcile(raw: Path, report: Path, bars: Path, output: Path) -> dict:
    manifest = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    with (report / "quote_diagnostics.csv").open(encoding="utf-8", newline="") as stream:
        derived = {(r["date"], r["symbol"]): r for r in csv.DictReader(stream)}
    with bars.open(encoding="utf-8", newline="") as stream:
        futures = {(r["date"], r["contract"]): float(r["settle"]) for r in csv.DictReader(stream)}
    checks, differences = [], []

    def check(name, passed, date="", symbol=""):
        item = {"date": date, "symbol": symbol, "check": name, "passed": bool(passed)}
        checks.append(item)
        if not passed:
            differences.append(item)

    def number(value):
        return float(value.replace(",", "")) if value not in {"", "-", "--", "—"} else None

    def same(actual, expected):
        return (not actual and expected is None) or (
            expected is not None
            and bool(actual)
            and math.isclose(float(actual), expected, abs_tol=1e-12, rel_tol=1e-12)
        )

    for receipt in manifest["records"]:
        if receipt.get("file"):
            content = (raw / receipt["file"]).read_bytes()
            check(
                "raw_receipt_hash",
                hashlib.sha256(content).hexdigest() == receipt["sha256"],
                receipt["id"],
            )
        if receipt["mode"] == "document" or receipt["status"] != "http_ok":
            continue
        html = content.decode("utf-8-sig")
        html = re.sub(r"<!--.*?-->", "", html, flags=re.S)
        tables = re.findall(r"<table\b[^>]*>(.*?)</table>", html, re.S | re.I)
        selected = []
        for table in tables:
            rows = []
            for tr in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table, re.S | re.I):
                rows.append(
                    [
                        unescape(re.sub("<[^>]+>", "", td)).strip()
                        for td in re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", tr, re.S | re.I)
                    ]
                )
            if rows and "合约" in rows[0] and "收盘价" in rows[0]:
                selected.append(rows)
        date = receipt["date"]
        check("one_quote_table", len(selected) == 1, date)
        if len(selected) != 1:
            continue
        rows = selected[0]
        offset = 1 if receipt["mode"] == "dated_query" else 0
        check("header_close_location", rows[0][offset + 4] == "收盘价", date)
        if offset:
            check("all_explicit_row_dates", all(row[0] == date for row in rows[1:]), date)
        else:
            year, month, day = map(int, date.split("-"))
            check(
                "article_heading_date",
                f"上海黄金交易所{year}年{month}月{day}日交易行情" in html,
                date,
            )
        lookup = {row[offset]: row for row in rows[1:]}
        for symbol in ("Au99.95", "Au99.99", "Ag99.99"):
            result = derived[(date, symbol)]
            row = lookup.get(symbol)
            raw_close = number(row[offset + 4]) if row else None
            raw_volume = number(row[offset + 8]) if row else None
            check("close_matches_raw", same(result["close"], raw_close), date, symbol)
            check(
                "volume_matches_raw", same(result["volume_as_published"], raw_volume), date, symbol
            )
            check(
                "strict_factor_gated",
                result["strict_spot_basis"] == "" and result["claim_eligible"] == "False",
                date,
                symbol,
            )
            check(
                "unknown_clock_not_backdated",
                result["publisher_available_at"] == ""
                and result["publisher_clock_verified"] == "False",
                date,
                symbol,
            )
            if result["diagnostic_available"] == "True":
                settle = futures.get((date, result["front_contract"]))
                check(
                    "settlement_matches_dated_contract_bar",
                    settle is not None and same(result["reference_settle"], settle),
                    date,
                    symbol,
                )
                expected = math.log(raw_close / settle) if settle else None
                check(
                    "log_ratio_independent_arithmetic",
                    same(result["diagnostic_log_quote_ratio"], expected),
                    date,
                    symbol,
                )
            else:
                check(
                    "unusable_quote_has_no_diagnostic",
                    result["diagnostic_log_quote_ratio"] == "",
                    date,
                    symbol,
                )
        check(
            "no_deferred_or_international_substitution",
            all(s in {"Au99.95", "Au99.99", "Ag99.99"} for d, s in derived if d == date),
            date,
        )
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "kind": "independent_spot_source_arithmetic_audit",
        "checks": len(checks),
        "differences": len(differences),
        "independent_statistical_samples": False,
        "historical_release_certified": False,
        "source_hashes": {
            "raw_manifest": hashlib.sha256((raw / "manifest.json").read_bytes()).hexdigest(),
            "diagnostics": hashlib.sha256(
                (report / "quote_diagnostics.csv").read_bytes()
            ).hexdigest(),
            "bars": hashlib.sha256(bars.read_bytes()).hexdigest(),
            "audit_code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    with (output / "checks.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["date", "symbol", "check", "passed"])
        writer.writeheader()
        writer.writerows(checks)
    (output / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "index.html").write_text(
        f"""<!doctype html><meta charset="utf-8"><title>现货独立对账</title><style>body{{font:17px/1.8 system-ui;max-width:900px;margin:50px auto;padding:24px;color:#193348;background:#f4f7fa}}</style><h1>现货原表与算术独立核对</h1><p>{len(checks)} 项检查，{len(differences)} 项差异。</p><p>独立读取原 HTML 的日期、收盘和成交量，核对选定期货的逐合约结算价及对数比值；未导入生产解析器或因子函数。零差异只表示核对范围内一致，不能认证初发版本、可交易性或统计显著。</p><p><a href="checks.csv">全部程序检查</a> · <a href="audit.json">来源哈希</a></p><pre>{escape(json.dumps(differences, ensure_ascii=False, indent=2))}</pre>""",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "report", "bars", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.raw, args.report, args.bars, args.output), indent=2))
