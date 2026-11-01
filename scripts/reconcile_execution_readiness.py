"""Independent source and arithmetic checker; no production source/parser/ledger imports."""

import argparse
import csv
import hashlib
import json
import re
import tomllib
import unicodedata
from datetime import date
from html import unescape
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Checks:
    def __init__(self):
        self.count = 0
        self.differences = []

    def test(self, name, passed):
        self.count += 1
        if not passed:
            self.differences.append(name)

    def close(self, name, a, b):
        self.test(name, bool(np.isclose(a, b, equal_nan=True, rtol=1e-12, atol=1e-14)))


def text(raw):
    cleaned = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>", "", raw.decode("utf-8-sig"), flags=re.S | re.I
    )
    return re.sub(
        r"\s+", "", unicodedata.normalize("NFKC", unescape(re.sub(r"<[^>]+>", "", cleaned)))
    )


def check_margin_demo(directory, checks):
    daily = pd.read_csv(directory / "daily.csv")
    trades = pd.read_csv(directory / "trades.csv")
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    checks.test("margin example synthetic", run["synthetic"] is True)
    for field, expected in {
        "equity": [999.0, 998.0, 998.0],
        "opening_reserved_margin": [100.0, 0.0, 0.0],
        "reserved_margin": [300.0, 0.0, 0.0],
        "fees": [1.0, 1.0, 0.0],
        "overnight_pnl": [0.0, 0.0, 0.0],
        "intraday_pnl": [0.0, 0.0, 0.0],
    }.items():
        checks.test(
            "margin demo " + field, np.allclose(daily[field], expected, atol=1e-12, rtol=1e-12)
        )
    checks.test("margin call only on first close", list(daily.margin_call) == [True, False, False])
    checks.test(
        "next opening margin liquidation",
        list(trades.reason) == ["rebalance", "margin_liquidation"],
    )
    checks.test("cash sums", np.allclose(1000.0 - daily.fees.cumsum(), daily.equity))
    checks.test("margin demo zero reconciliation", daily.reconciliation_error.eq(0).all())


def reconcile(config, report, output, margin_demo=None):
    checks = Checks()
    plan = tomllib.loads(config.read_text(encoding="utf-8"))
    root = config.resolve().parent.parent
    raw = root / plan["paths"]["raw"]
    manifest = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((report / "run.json").read_text(encoding="utf-8"))
    checks.test(
        "protocol receipt", sha(config) == manifest["protocol_sha256"] == summary["protocol_sha256"]
    )
    checks.test(
        "capture receipt hash", sha(raw / "manifest.json") == summary["capture_manifest_sha256"]
    )
    for name, expected in plan["input_hashes"].items():
        checks.test(name + " frozen input", sha(root / plan["paths"][name]) == expected)
    for name, expected in summary["code_sha256"].items():
        checks.test(name + " code hash", sha(root / "src/commodity_factor_lab" / name) == expected)
    parameters = pd.read_csv(report / "snapshot_parameters.csv", dtype={"raw_trade_fee_unit": str})
    source_audit = pd.read_csv(report / "source_audit.csv")
    rules = json.loads((report / "notice_rules.json").read_text(encoding="utf-8"))
    bodies = {}
    clock_after_day = 0
    for capture in manifest["records"]:
        key = capture["id"]
        path = raw / capture["file"]
        checks.test(key + " raw hash", sha(path) == capture["sha256"])
        saved = source_audit.loc[source_audit.id.eq(key)].iloc[0]
        checks.test(key + " audit hash", saved.sha256 == capture["sha256"])
        checks.test(key + " audit url", saved.url == capture["url"])
        if capture["kind"] in {"notice", "current_hours"}:
            bodies[key] = text(path.read_bytes())
            if capture["kind"] == "notice":
                header_date = re.search(r"发布日期[:：](\d{4}-\d{2}-\d{2})", bodies[key])[1]
                checks.test(
                    key + " declared date", rules[key]["source_declared_date"] == header_date
                )
                checks.test(key + " no first vintage", not rules[key]["first_vintage_certified"])
            continue
        body = json.loads(path.read_text(encoding="utf-8-sig"))
        checks.test(key + " report date", str(body["report_date"]) == capture["date"])
        stamp = pd.to_datetime(body["update_date"], format="%Y%m%d %H:%M:%S").tz_localize(
            "Asia/Shanghai"
        )
        late = stamp.date() > pd.Timestamp(capture["date"]).date()
        clock_after_day += late
        checks.test(key + " source update clock", saved.source_update_at == stamp.isoformat())
        checks.test(key + " late clock flag", bool(saved.source_update_after_report_date) == late)
        collection = body[
            "Settlement" if capture["kind"] == "settlement" else "ContractDailyTradeArgument"
        ]
        lookup = {row["INSTRUMENTID"].strip().lower(): row for row in collection}
        selected = parameters.loc[parameters.source_id.eq(key)]
        checks.test(key + " selected unique", selected.contract.is_unique)
        for item in selected.itertuples():
            original = lookup[item.contract]
            checks.test(
                key + item.contract + " date",
                item.date == f"{capture['date'][:4]}-{capture['date'][4:6]}-{capture['date'][6:]}",
            )
            checks.test(
                key + item.contract + " row date", str(original["TRADINGDAY"]) == capture["date"]
            )
            checks.close(
                key + item.contract + " long margin",
                item.spec_long_margin_rate,
                float(original["SPEC_LONGMARGINRATIO"]),
            )
            checks.close(
                key + item.contract + " short margin",
                item.spec_short_margin_rate,
                float(original["SPEC_SHORTMARGINRATIO"]),
            )
            if capture["kind"] == "settlement":
                checks.close(
                    key + item.contract + " raw fee field",
                    item.raw_trade_fee_ratio,
                    float(original["TRADEFEERATION"]),
                )
                checks.test(
                    key + item.contract + " raw fee unit",
                    item.raw_trade_fee_unit == str(original["TRADEFEEUNIT"]),
                )
    checks.test("all parameter rows", len(parameters) == 22)
    checks.test("all source audit rows", len(source_audit) == 16)
    checks.test("all captures retained", len(manifest["records"]) == 16)
    checks.test("day-late sources", clock_after_day == summary["source_update_after_report_date"])
    checks.test("May reduced rate source words", "万分之零点二" in bodies["rb_offcycle_may"])
    checks.test(
        "May excluded months source words", "除1月、5月、10月合约" in bodies["rb_offcycle_may"]
    )
    checks.test(
        "May delivery rule source words",
        "交割月前第二月的第一个交易日起" in bodies["rb_offcycle_may"],
    )
    checks.test(
        "May regular rate source words", "恢复到成交金额的万分之一" in bodies["rb_offcycle_may"]
    )
    checks.close(
        "extracted reduced rate", rules["rb_offcycle_may"]["reduced_fee_rate"], 0.2 / 10000
    )
    checks.close(
        "extracted restored rate", rules["rb_offcycle_may"]["restored_fee_rate"], 1.0 / 10000
    )
    checks.test(
        "same-contract April source",
        "RB2410合约交易手续费调整为成交金额的万分之一" in bodies["rb2410_april"],
    )
    calendar = pd.read_csv(root / plan["paths"]["calendar"])
    sessions = sorted(date.fromisoformat(day) for day in calendar.date)
    fees = pd.read_csv(report / "fee_cases.csv")
    for fee in fees.itertuples():
        month = int(fee.contract[-2:])
        day = date.fromisoformat(fee.date)
        expected, reset = np.nan, None
        if fee.contract == "rb2410":
            expected = 0.0001
        elif day >= date(2024, 5, 31):
            reset_month = month - 2
            reset = min(
                slot for slot in sessions if slot.year == 2024 and slot.month == reset_month
            )
            expected = 0.0001 if day >= reset else 0.00002
        checks.close(
            fee.date + fee.contract + " notice candidate", fee.notice_expected_rate, expected
        )
        checks.test(
            fee.date + fee.contract + " maturity reset",
            (reset is None and pd.isna(fee.maturity_reset_date))
            or (reset is not None and fee.maturity_reset_date == reset.isoformat()),
        )
        checks.test(
            fee.date + fee.contract + " remains unadmitted", not fee.historical_execution_eligible
        )
        snapshot = parameters.loc[
            parameters.kind.eq("settlement")
            & parameters.date.eq(fee.date)
            & parameters.contract.eq(fee.contract)
        ]
        if len(snapshot):
            checks.close(
                fee.date + fee.contract + " snapshot field",
                fee.raw_trade_fee_ratio,
                snapshot.iloc[0].raw_trade_fee_ratio,
            )
            checks.test(
                fee.date + fee.contract + " equality label",
                fee.snapshot_matches_candidate_numerically
                == np.isclose(expected, snapshot.iloc[0].raw_trade_fee_ratio),
            )
        else:
            checks.test(
                fee.date + fee.contract + " no invented comparison",
                pd.isna(fee.snapshot_matches_candidate_numerically),
            )
    phases = pd.read_csv(report / "margin_phases.csv")
    for row in phases.itertuples():
        for kind in ("trading", "settlement"):
            original = parameters.loc[
                parameters.date.eq(row.date)
                & parameters.contract.eq(row.contract)
                & parameters.kind.eq(kind)
            ].iloc[0]
            checks.close(
                row.date + row.contract + kind + " margin phase",
                getattr(row, f"spec_long_margin_rate_{kind}"),
                original.spec_long_margin_rate,
            )
            checks.close(
                row.date + row.contract + kind + " short margin phase",
                getattr(row, f"spec_short_margin_rate_{kind}"),
                original.spec_short_margin_rate,
            )
        checks.test(
            row.date + row.contract + " phase difference",
            row.observed_phase_rate_diff
            == (row.spec_long_margin_rate_trading != row.spec_long_margin_rate_settlement),
        )
        checks.test(
            row.date + row.contract + " no automatic restore",
            not row.automatic_holiday_restoration_performed,
        )
    checks.test(
        "conditional restoration words retained",
        "第一个未出现单边市的交易日收盘结算时" in bodies["cny2025"],
    )
    checks.test(
        "conditional restoration extracted", rules["cny2025"]["restoration_conditional"] is True
    )
    for code, phrase in (
        ("cu", "投机交易保证金比例调整为12%"),
        ("rb", "投机交易保证金比例调整为10%"),
    ):
        checks.test(code + " holiday margin words", phrase in bodies["cny2025"])
        checks.close(
            code + " holiday margin extraction",
            rules["cny2025"]["rates"][code]["spec_margin"],
            0.12 if code == "cu" else 0.1,
        )
        checks.close(
            code + " holiday hedge margin extraction",
            rules["cny2025"]["rates"][code]["hedge_margin"],
            0.11 if code == "cu" else 0.09,
        )
        checks.close(
            code + " holiday price limit extraction",
            rules["cny2025"]["rates"][code]["price_limit"],
            0.1 if code == "cu" else 0.08,
        )
    schedule = pd.read_csv(report / "session_evidence.csv")
    checks.test("no invented actual clocks", schedule.actual_open_at.isna().all())
    checks.test(
        "all schedule certification flags false",
        not schedule.historical_actual_opening_certified.any(),
    )
    checks.test(
        "April trading day versus Friday night",
        rules["rb2410_april"]["effective_trading_date"] == "2024-04-01"
        and rules["rb2410_april"]["effective_night_natural_date"] == "2024-03-29",
    )
    checks.test(
        "May trading day versus Thursday night",
        rules["rb_offcycle_may"]["effective_trading_date"] == "2024-05-31"
        and rules["rb_offcycle_may"]["effective_night_natural_date"] == "2024-05-30",
    )
    gates = pd.read_csv(report / "admission_gates.csv")
    checks.test(
        "seven separate unpassed gates",
        len(gates) == 7 and not gates.satisfied.any() and not gates.real_execution_eligible.any(),
    )
    checks.test("no new forecasts", summary["prediction_tests_added"] == 0)
    checks.test(
        "fifteen exploration items retained",
        summary["registered_exploration_family_size_unchanged"] == 15,
    )
    checks.test("no actual execution qualification", summary["real_execution_eligible"] is False)
    if margin_demo:
        check_margin_demo(margin_demo, checks)
    result = {
        "kind": "independent_execution_source_and_phase_reconciliation",
        "checks": checks.count,
        "differences": len(checks.differences),
        "production_modules_imported": False,
        "independent_statistical_samples": False,
        "historical_first_vintage_certified": False,
        "audit_code_sha256": sha(__file__),
        "run_sha256": sha(report / "run.json"),
        "capture_manifest_sha256": sha(raw / "manifest.json"),
        "report_csv_sha256": {p.name: sha(p) for p in sorted(report.glob("*.csv"))},
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "differences.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["check"])
        writer.writerows([[name] for name in checks.differences])
    (output / "index.html").write_text(
        f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>执行来源独立核对</title><body style="max-width:900px;margin:45px auto;padding:24px;font:17px/1.9 system-ui"><h1>执行来源与两阶段保证金独立核对</h1><p>{checks.count}项检查，{len(checks.differences)}处差异。</p><p>独立读取原HTML、JSON、公告日期、费率单位、合同月份和日历；逐项核对来源指纹、快照字段、候选费率、交易/结算保证金及准入门槛。不调用生产解析或资金账代码；合成例子另按手算现金和保证金核对。检查次数不代表独立统计样本，也不认证历史初发或可成交性。</p><p><a href="audit.json">指纹与检查记录</a> · <a href="differences.csv">差异表</a></p></body></html>',
        encoding="utf-8",
    )
    print(json.dumps({"checks": checks.count, "differences": len(checks.differences)}))
    return not checks.differences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/execution_readiness.toml"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin-demo", type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if reconcile(args.config, args.report, args.output, args.margin_demo) else 1)
