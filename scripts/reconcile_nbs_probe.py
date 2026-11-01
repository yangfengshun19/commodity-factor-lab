"""Independent NBS raw-table and event checks; no production parser/factor imports."""

import argparse
import csv
import hashlib
import json
import math
import re
import tomllib
import unicodedata
from datetime import datetime, timedelta, timezone
from html import escape, unescape
from pathlib import Path


def reconcile(
    raw: Path, report: Path, calendar: Path, output: Path, rules: Path | None = None
) -> dict:
    def read(path):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def norm(value):
        compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))
        return re.sub(r"注\d+", "", compact).replace("—", "-").replace("–", "-").replace("Φ", "φ")

    def plain(html):
        html = re.sub(r"<!--.*?-->", "", html, flags=re.S)
        html = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", html, flags=re.S | re.I)
        return unescape(re.sub("<[^>]+>", " ", html))

    def numerical_equal(actual, expected):
        expected = norm(expected).replace(",", "")
        return (not actual and expected in {"", "-", "--"}) or (
            bool(actual)
            and expected not in {"", "-", "--"}
            and math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)
        )

    manifest = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    prices = {
        (row["period"], row["product"]): row
        for row in read(report / "published_reference_prices.csv")
    }
    specifications = {
        (row["period"], row["number"]): row for row in read(report / "product_specifications.csv")
    }
    events = {row["period"]: row for row in read(report / "release_events.csv")}
    sessions = read(calendar)
    checks = []

    def check(name, passed, period="", product=""):
        checks.append({"period": period, "product": product, "check": name, "passed": bool(passed)})

    for item in manifest["records"]:
        content = (raw / item["file"]).read_bytes()
        check("raw_receipt_hash", hashlib.sha256(content).hexdigest() == item["sha256"], item["id"])
        if item["kind"] != "release":
            continue
        period = item.get("period", item["id"])
        html = content.decode("utf-8-sig")
        no_comments = re.sub(r"<!--.*?-->", "", html, flags=re.S)
        tables = []
        for table in re.findall(r"<table\b[^>]*>(.*?)</table>", no_comments, re.S | re.I):
            tables.append(
                [
                    [
                        norm(plain(cell))
                        for cell in re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row, re.S | re.I)
                    ]
                    for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table, re.S | re.I)
                ]
            )
        quote_tables = [
            tab
            for tab in tables
            if tab
            and len(tab[0]) >= 3
            and tab[0][:2] == ["产品名称", "单位"]
            and re.fullmatch(r"本期价格\d*\(元\)\d*", tab[0][2])
        ]
        spec_tables = [
            tab for tab in tables if tab and tab[0][:3] == ["序号", "监测产品", "规格型号"]
        ]
        check(
            "consistent_quote_copies",
            bool(quote_tables) and all(tab == quote_tables[0] for tab in quote_tables),
            period,
        )
        check(
            "consistent_specification_copies",
            bool(spec_tables) and all(tab == spec_tables[0] for tab in spec_tables),
            period,
        )
        header = plain(html[: html.lower().find("<table")])
        clock = re.findall(r"20\d{2}[/-]\d{2}[/-]\d{2}\s+\d{2}:\d{2}", header)
        check("one_displayed_publication_clock", len(set(clock)) == 1, period)
        source_clock = (
            datetime.strptime(re.sub(r"\s+", " ", clock[0]).replace("/", "-"), "%Y-%m-%d %H:%M")
            .replace(tzinfo=timezone(timedelta(hours=8)))
            .astimezone(timezone.utc)
        )
        rows = [row for row in quote_tables[0][1:] if len(row) == 5 and row[1] in {"吨", "千克"}]
        check("fifty_source_products_once", len(rows) == 50, period)
        source_text = norm(plain(html))
        scope = (
            "24_provinces"
            if "据对24个省" in source_text
            else "nationwide_31_provinces"
            if "监测范围涵盖全国31个省" in source_text
            else "scope_not_verified"
        )
        spec_lookup = {
            norm(row[1] + "(" + row[2] + ")"): row[3]
            for row in spec_tables[0][1:]
            if len(row) == 4 and row[0].isdigit()
        }
        for cells in rows:
            derived = prices[(period, cells[0])]
            check(
                "price_matches_raw",
                numerical_equal(derived["reference_price"], cells[2]),
                period,
                cells[0],
            )
            check(
                "quantity_unit_matches_raw", derived["quantity_unit"] == cells[1], period, cells[0]
            )
            description = spec_lookup.get(cells[0], "")
            key = hashlib.sha256(
                "|".join((cells[0], cells[1], scope, description)).encode()
            ).hexdigest()
            check(
                "scope_specification_and_identity_match_raw",
                derived["monitoring_scope"] == scope
                and derived["specification_description"] == description
                and derived["comparability_key"] == key,
                period,
                cells[0],
            )
            check(
                "declared_clock_matches_header",
                datetime.fromisoformat(derived["source_declared_published_at"]) == source_clock,
                period,
                cells[0],
            )
            check(
                "reference_not_execution_or_certified_vintage",
                derived["executable_quote"]
                == derived["historical_first_vintage_certified"]
                == derived["strict_spot_basis_eligible"]
                == "False",
                period,
                cells[0],
            )
        for row in spec_tables[0][1:]:
            if row and row[0].isdigit():
                derived = specifications[(period, row[0])]
                check(
                    "specification_matches_raw",
                    [derived["product"], derived["specification"], derived["description"]]
                    == row[1:],
                    period,
                    row[1],
                )
        next_session = next(
            (
                row
                for row in sessions
                if datetime.fromisoformat(row["signal_cutoff"]) >= source_clock
            ),
            None,
        )
        previous = [
            row for row in sessions if datetime.fromisoformat(row["signal_cutoff"]) < source_clock
        ]
        check(
            "first_signal_not_before_publication",
            events[period]["first_signal_date"][:10]
            == (next_session["date"][:10] if next_session else ""),
            period,
        )
        check(
            "previous_signal_is_strictly_earlier",
            events[period]["previous_signal_date"][:10]
            == (previous[-1]["date"][:10] if previous else ""),
            period,
        )
    check(
        "cancelled_periods_not_reconstructed",
        not any(period in {"2024-02-middle", "2025-01-late"} for period, _ in prices),
    )
    if rules is not None:
        cancellations = tomllib.loads(rules.read_text(encoding="utf-8"))["cancellations"]
        discovery = json.loads((raw.parent / "discovery/manifest.json").read_text(encoding="utf-8"))
        sources = {item["id"]: item for item in discovery["records"]}
        ledger = read(report / "period_ledger.csv")
        expected = {
            f"{year}-{month:02d}-{part}"
            for year in range(2020, 2026)
            for month in range(1, 13)
            for part in ["early", "middle", "late"]
        }
        check(
            "all_216_calendar_periods_once",
            len(ledger) == 216 and {row["period"] for row in ledger} == expected,
        )
        cancelled = {item["period"] for item in cancellations}
        check(
            "ledger_exactly_matches_all_cancellation_rules",
            {row["period"] for row in ledger if row["status"] == "officially_cancelled"}
            == cancelled,
        )
        for item in cancellations:
            source = sources[item["schedule_id"]]
            content = (raw.parent / "discovery" / source["file"]).read_bytes()
            check(
                "official_cancellation_evidence",
                hashlib.sha256(content).hexdigest() == item["source_sha256"]
                and source["url"] == item["source_url"]
                and norm(item["evidence_fragment"]) in norm(plain(content.decode("utf-8"))),
                item["period"],
            )
            check(
                "cancelled_period_has_no_reconstructed_prices",
                not any(period == item["period"] for period, _ in prices),
                item["period"],
            )
    differences = [item for item in checks if not item["passed"]]
    result = {
        "kind": "independent_nbs_source_and_event_reconciliation",
        "checks": len(checks),
        "differences": len(differences),
        "independent_statistical_samples": False,
        "historical_first_vintage_certified": False,
        "inputs": {
            "raw_manifest": hashlib.sha256((raw / "manifest.json").read_bytes()).hexdigest(),
            "prices": hashlib.sha256(
                (report / "published_reference_prices.csv").read_bytes()
            ).hexdigest(),
            "events": hashlib.sha256((report / "release_events.csv").read_bytes()).hexdigest(),
            "calendar": hashlib.sha256(calendar.read_bytes()).hexdigest(),
            "audit_code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    if rules is not None:
        result["inputs"]["cancellation_rules"] = hashlib.sha256(rules.read_bytes()).hexdigest()
        result["inputs"]["period_ledger"] = hashlib.sha256(
            (report / "period_ledger.csv").read_bytes()
        ).hexdigest()
        result["inputs"]["discovery_manifest"] = hashlib.sha256(
            (raw.parent / "discovery/manifest.json").read_bytes()
        ).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "checks.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["period", "product", "check", "passed"])
        writer.writeheader()
        writer.writerows(checks)
    (output / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "index.html").write_text(
        f"""<!doctype html><meta charset="utf-8"><title>工业品原表独立核对</title><style>body{{font:17px/1.8 system-ui;max-width:900px;margin:40px auto;padding:24px;color:#193348;background:#f4f7fa}}</style><h1>原始表格与公告时点独立核对</h1><p>{len(checks)}项检查，{len(differences)}项差异。</p><p>另用文本表格读取器核对原价格、单位、产品规格、正文发布时间及首个期货信号日期，未导入生产解析器或因子函数。零差异仅表示核对范围内一致，不能替代初发版本、可交易性或预测效果认证。</p><p><a href="checks.csv">全部检查</a> · <a href="audit.json">来源与源码哈希</a></p><pre>{escape(json.dumps(differences, ensure_ascii=False, indent=2))}</pre>""",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "report", "calendar", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rules", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            reconcile(args.raw, args.report, args.calendar, args.output, args.rules), indent=2
        )
    )
