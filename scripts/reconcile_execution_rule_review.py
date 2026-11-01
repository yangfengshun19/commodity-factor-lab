"""Independent checks of rule dates, raw margin fields, component arithmetic and auction timing."""

import argparse
import csv
import hashlib
import json
import re
import tomllib
import unicodedata
from html import unescape
from pathlib import Path

import numpy as np
import pandas as pd


class Checks:
    def __init__(self):
        self.count, self.differences = 0, []

    def test(self, name, passed):
        self.count += 1
        if not passed:
            self.differences.append(name)

    def close(self, name, actual, expected):
        self.test(name, bool(np.isclose(actual, expected, atol=1e-14, rtol=1e-12, equal_nan=True)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(path):
    raw = Path(path).read_text(encoding="utf-8-sig")
    raw = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", raw, flags=re.S | re.I)
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", unescape(re.sub(r"<[^>]+>", "", raw))))


def check_scenarios(parameters, scenarios, components, calendar, rules, checks):
    checks.test("48 scenario rows", len(scenarios) == 48)
    checks.test(
        "scenario keys unique",
        not scenarios.duplicated(["date", "phase", "contract", "branch"]).any(),
    )
    checks.test("24 parameter rows", len(parameters) == 24)
    inputs = parameters.set_index(["date", "kind", "contract"])
    reference = (
        parameters.loc[parameters.date.eq("2025-01-24") & parameters.kind.eq("trading")]
        .set_index("contract")
        .spec_long_margin_rate
    )
    expected_components = []
    for row in scenarios.itertuples():
        date = pd.Timestamp(row.date)
        key = (row.date, row.phase, row.contract)
        original = inputs.loc[key]
        code = row.contract[:2]
        delivery = pd.Timestamp(2000 + int(row.contract[2:4]), int(row.contract[4:]), 1)
        target_month = delivery.to_period("M") - 1
        dates = pd.to_datetime(calendar.date)
        boundary = dates.loc[dates.dt.to_period("M").eq(target_month)].min()
        checks.test(str(key) + " maturity boundary", row.maturity_boundary == str(boundary.date()))
        floors = {
            "product_listed_floor": float(rules[code]["listed_floor"]),
            "observed_preholiday_reference": float(reference[row.contract]),
        }
        if date >= boundary:
            floors["month_before_delivery_floor"] = float(rules[code]["month_before_floor"])
        active = date > pd.Timestamp("2025-01-24") or (
            date == pd.Timestamp("2025-01-24") and row.phase == "settlement"
        )
        reopened = date > pd.Timestamp("2025-02-05") or (
            date == pd.Timestamp("2025-02-05") and row.phase == "settlement"
        )
        if row.branch == "released_at_reopening_settlement" and reopened:
            active = False
        if active:
            floors["holiday_spec_floor"] = 0.12 if code == "cu" else 0.1
        expected = max(floors.values())
        checks.close(
            str(key) + row.branch + " maximum not sum", row.known_component_floor, expected
        )
        checks.close(
            str(key) + row.branch + " observed long",
            row.snapshot_spec_long_margin_rate,
            original.spec_long_margin_rate,
        )
        checks.close(
            str(key) + row.branch + " observed short",
            row.snapshot_spec_short_margin_rate,
            original.spec_short_margin_rate,
        )
        checks.test(
            str(key) + row.branch + " scenario matching flag",
            row.matches_observed_long_rate == np.isclose(expected, original.spec_long_margin_rate),
        )
        checks.test(
            str(key) + row.branch + " holiday state", row.holiday_active_in_scenario == active
        )
        checks.test(
            str(key) + row.branch + " reference rows labelled",
            row.reference_calibration_row == (row.date == "2025-01-24" and row.phase == "trading"),
        )
        checks.test(
            str(key) + row.branch + " unresolved exact rate", pd.isna(row.fully_resolved_rate)
        )
        checks.test(
            str(key) + row.branch + " no execution promotion", not row.real_execution_eligible
        )
        checks.test(
            str(key) + row.branch + " incomplete status",
            row.status == "incomplete_components_lower_bound_only",
        )
        binding = ";".join(
            name
            for name, value in floors.items()
            if np.isclose(value, expected, atol=1e-14, rtol=1e-12)
        )
        checks.test(
            str(key) + row.branch + " binding components", row.binding_components == binding
        )
        selected = components.loc[
            components.date.eq(row.date)
            & components.phase.eq(row.phase)
            & components.contract.eq(row.contract)
            & components.branch.eq(row.branch)
        ]
        checks.test(
            str(key) + row.branch + " all components",
            set(selected.component) == set(floors) and len(selected) == len(floors),
        )
        for component in selected.itertuples():
            checks.close(
                str(key) + row.branch + component.component,
                component.absolute_floor,
                floors[component.component],
            )
            expected_components.append(
                (row.date, row.phase, row.contract, row.branch, component.component)
            )
    checks.test("no extra component rows", len(expected_components) == len(components))


def check_auction_files(directory, checks):
    frame = pd.read_csv(directory / "auction_cutoff_checks.csv")
    checks.test("three synthetic deadline cases", len(frame) == 3)
    for row in frame.itertuples():
        decision, cutoff, opening = map(
            pd.Timestamp, (row.decision_at, row.order_cutoff_at, row.open_at)
        )
        checks.test(row.decision_at + " artificial input", row.synthetic)
        checks.test(
            row.decision_at + " strict decision deadline", row.rejected == (decision >= cutoff)
        )
        checks.test(row.decision_at + " before opening", decision < opening)
    daily = pd.read_csv(directory / "daily.csv")
    checks.test("margin synthetic cash unchanged", list(daily.equity) == [999.0, 998.0, 998.0])
    checks.test(
        "margin synthetic opening occupation",
        list(daily.opening_reserved_margin) == [100.0, 0.0, 0.0],
    )
    checks.test(
        "margin synthetic settlement occupation", list(daily.reserved_margin) == [300.0, 0.0, 0.0]
    )
    checks.test("cash reconciliation zero", daily.reconciliation_error.eq(0).all())


def reconcile(config, report, output, demo=None):
    plan = tomllib.loads(config.read_text(encoding="utf-8"))
    root = config.resolve().parent.parent
    paths = {key: root / value for key, value in plan["paths"].items()}
    run = json.loads((report / "run.json").read_text(encoding="utf-8"))
    captures = json.loads((paths["raw"] / "manifest.json").read_text(encoding="utf-8"))
    checks = Checks()
    checks.test(
        "protocol hashes", run["protocol_sha256"] == captures["protocol_sha256"] == sha(config)
    )
    checks.test(
        "capture fingerprint", run["capture_manifest_sha256"] == sha(paths["raw"] / "manifest.json")
    )
    for name, expected in plan["input_hashes"].items():
        checks.test(name + " frozen input", sha(paths[name]) == expected)
    for name, expected in run["code_sha256"].items():
        checks.test(
            name + " code fingerprint", sha(root / "src/commodity_factor_lab" / name) == expected
        )
    rules = json.loads((report / "rules.json").read_text(encoding="utf-8"))
    versions = pd.read_csv(report / "source_versions.csv")
    bodies = {}
    expected_sources = {source["id"]: source for source in plan["sources"]}
    for item in captures["records"]:
        name = item["id"]
        checks.test(
            name + " frozen identity",
            all(item.get(key) == value for key, value in expected_sources[name].items()),
        )
        path = paths["raw"] / item["file"]
        checks.test(name + " raw fingerprint", sha(path) == item["sha256"])
        declared = versions.loc[versions.id.eq(name)].iloc[0]
        checks.test(name + " saved fingerprint", declared.sha256 == item["sha256"])
        bodies[name] = text(path)
        if item["kind"] == "rule":
            pub = re.search(r"发布日期[:：](\d{4}-\d{2}-\d{2})", bodies[name])[1]
            dates = {
                f"{int(y):04}-{int(m):02}-{int(d):02}"
                for y, m, d in re.findall(r"自(\d{4})年(\d+)月(\d+)日起(?:开始)?实施", bodies[name])
            }
            checks.test(name + " raw publication date", declared.source_declared_date == pub)
            checks.test(name + " raw effective date", dates == {declared.declared_effective_date})
    checks.test("parent release date", "自2024年10月23日实施" in bodies["rule_release_20240823"])
    checks.test(
        "risk maximum rule",
        "交易保证金按照规定交易保证金比例中的最高值收取" in bodies["risk_rules_202409"],
    )
    for key, source in (
        ("cu", "cu_rules_202409"),
        ("rb", "rb_rules_202409"),
        ("later_cu_negative", "cu_rules_later_negative"),
    ):
        body = bodies[source]
        version = versions.loc[versions.id.eq(source)].iloc[0]
        checks.test(
            key + " product rule publication",
            rules[key]["source_declared_date"] == version.source_declared_date,
        )
        checks.test(
            key + " product rule effective date",
            rules[key]["declared_effective_date"] == version.declared_effective_date,
        )
        for field, label in (
            ("listed_floor", "合约挂牌之日起"),
            ("month_before_floor", "交割月前第一月的第一个交易日起"),
            ("delivery_month_floor", "交割月份第一个交易日起"),
            ("last_two_sessions_floor", "最后交易日前二个交易日起"),
        ):
            expected = int(re.search(re.escape(label) + r"(\d+)%", body)[1]) / 100
            checks.close(key + field + " rule table", rules[key][field], expected)
        checks.close(
            key + " contract multiplier",
            rules[key]["multiplier_tonnes"],
            int(re.search(r"交易单位为(\d+)吨/手", body)[1]),
        )
    old = json.loads(paths["seed_manifest"].read_text(encoding="utf-8"))
    old_lookup = {item["id"]: item for item in old["records"]}
    parameters = pd.read_csv(report / "seed_parameters.csv")
    raw_lookups = {}
    for name in parameters.source_id.unique():
        source = old_lookup[name]
        path = paths["seed"] / source["file"]
        checks.test(name + " original input fingerprint", sha(path) == source["sha256"])
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        kind = "Settlement" if name.startswith("settlement") else "ContractDailyTradeArgument"
        raw_lookups[name] = {row["INSTRUMENTID"].strip().lower(): row for row in data[kind]}
    for row in parameters.itertuples():
        original = raw_lookups[row.source_id][row.contract]
        checks.close(
            row.source_id + row.contract + " long margin",
            row.spec_long_margin_rate,
            float(original["SPEC_LONGMARGINRATIO"]),
        )
        checks.close(
            row.source_id + row.contract + " short margin",
            row.spec_short_margin_rate,
            float(original["SPEC_SHORTMARGINRATIO"]),
        )
        checks.test(
            row.source_id + row.contract + " raw trading date",
            row.date.replace("-", "") == str(original["TRADINGDAY"]),
        )
    scenarios = pd.read_csv(report / "margin_scenarios.csv")
    components = pd.read_csv(report / "margin_components.csv")
    calendar = pd.read_csv(paths["calendar"])
    check_scenarios(parameters, scenarios, components, calendar, rules, checks)
    exclusions = pd.read_csv(report / "version_exclusions.csv")
    checks.test(
        "four later version dates rejected",
        len(exclusions) == 4 and not exclusions.date_applicable.any(),
    )
    checks.test(
        "later date truly after reviewed period",
        pd.Timestamp(rules["later_cu_negative"]["declared_effective_date"])
        > pd.Timestamp("2025-02-06"),
    )
    auction = pd.read_csv(report / "scheduled_auction.csv").iloc[0]
    checks.test(
        "holiday raw fingerprint",
        sha(paths["seed"] / old_lookup["cny2025"]["file"]) == old_lookup["cny2025"]["sha256"],
    )
    checks.test(
        "dated holiday auction",
        "08:55-09:00" in text(paths["seed"] / old_lookup["cny2025"]["file"]),
    )
    checks.test(
        "four-minute submission source", "集合竞价前4分钟" in bodies["trading_rules_202409"]
    )
    checks.test(
        "one-minute matching source", "后1分钟为集合竞价撮合" in bodies["trading_rules_202409"]
    )
    checks.test(
        "fallback first trade source",
        "以当日竞价交易的第一笔成交价为开盘价" in bodies["trading_rules_202409"],
    )
    checks.test(
        "scheduled submission end",
        pd.Timestamp(auction.scheduled_submission_cutoff_at)
        == pd.Timestamp("2025-02-05T08:59:00+08:00"),
    )
    checks.test(
        "no fabricated execution clock",
        pd.isna(auction.actual_open_at) and not auction.actual_execution_clock_certified,
    )
    checks.test(
        "zero exact rates",
        scenarios.fully_resolved_rate.isna().all() and run["fully_resolved_margin_rates"] == 0,
    )
    checks.test(
        "no independent confirmation claim",
        run["historical_first_vintage_certified"] is False
        and run["real_execution_eligible"] is False,
    )
    checks.test(
        "no new forecast experiment",
        run["prediction_tests_added"] == 0
        and run["registered_exploration_family_size_unchanged"] == 15,
    )
    if demo is not None:
        check_auction_files(demo, checks)
    result = {
        "kind": "independent_historical_rule_component_review",
        "checks": checks.count,
        "differences": len(checks.differences),
        "production_modules_imported": False,
        "independent_statistical_samples": False,
        "historical_first_vintage_certified": False,
        "audit_code_sha256": sha(__file__),
        "run_sha256": sha(report / "run.json"),
        "report_csv_sha256": {p.name: sha(p) for p in sorted(report.glob("*.csv"))},
        "rule_json_sha256": sha(report / "rules.json"),
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
        f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>历史规则独立复核</title><body style="font:17px/1.85 system-ui;max-width:950px;margin:40px auto;padding:24px"><h1>历史规则组件与竞价时序独立复核</h1><p>{checks.count}项检查，{len(checks.differences)}处差异。</p><p>另读HTML日期、规则表、JSON保证金和日历，独立重建比例组件与两种情景；核对晚版排除及计划竞价截止。无生产解析或资金账导入。情景与已观测值一致不等于初发认证、完整通知覆盖或可成交证明；检查数不是独立统计样本。</p><p><a href="audit.json">检查与指纹</a> · <a href="differences.csv">差异表</a></p></body></html>',
        encoding="utf-8",
    )
    print(json.dumps({"checks": checks.count, "differences": len(checks.differences)}))
    return not checks.differences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/execution_rule_review.toml"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demo", type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if reconcile(args.config, args.report, args.output, args.demo) else 1)
