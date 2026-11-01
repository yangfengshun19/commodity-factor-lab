import hashlib
import importlib.util
import json
import tempfile
import tomllib
import unittest
from pathlib import Path

import pandas as pd
from test_execution_readiness import notice, snapshot

from commodity_factor_lab.execution_rule_review import (
    date_applicable,
    known_margin_floor,
    month_before_start,
    product_rule,
    review_plan,
    run_rule_review,
    scheduled_auction,
)

ROOT = Path(__file__).resolve().parents[1]


def product(code="cu", later=False):
    publication, effective = (
        ("2025-08-07", "2025年8月8日") if later else ("2024-09-03", "2024年10月23日")
    )
    return f"<p>发布日期：{publication} 自{effective}起实施。交易单位为{5 if code == 'cu' else 10}吨/手。合约挂牌之日起5% 交割月前第一月的第一个交易日起10% 交割月份第一个交易日起15% 最后交易日前二个交易日起20%</p>".encode()


def trading_rule():
    return "<p>发布日期：2024-09-03 自2024年10月23日起实施。集合竞价前4分钟为合约买、卖指令申报时间，后1分钟为集合竞价撮合时间。集合竞价未产生成交价格的，以当日竞价交易的第一笔成交价为开盘价。</p>".encode()


def fixture(root):
    config_dir = root / "configs"
    config_dir.mkdir()
    (root / "input").mkdir()
    (root / "seed").mkdir()
    (root / "raw").mkdir()
    (root / "src/commodity_factor_lab").mkdir(parents=True)
    for name in ("execution_rule_review.py", "execution_sources.py", "execution.py"):
        (root / "src/commodity_factor_lab" / name).write_bytes(
            (ROOT / "src/commodity_factor_lab" / name).read_bytes()
        )
    text = (ROOT / "configs/execution_rule_review.toml").read_text(encoding="utf-8")
    original = tomllib.loads(text)
    dates = pd.bdate_range("2025-01-01", "2025-04-30")
    dates = dates[~dates.isin(pd.date_range("2025-01-28", "2025-02-04"))]
    pd.DataFrame({"date": dates}).to_csv(root / "input/calendar.csv", index=False)
    pd.DataFrame({"analysis": list(range(15))}).to_csv(root / "input/prior.csv", index=False)
    (root / "input/previous.json").write_text("{}", encoding="utf-8")
    links = {
        "links": [
            {"url": item["url"]}
            for item in original["sources"]
            if "/historicalversion/" in item["url"]
        ]
    }
    (root / "input/links.json").write_text(json.dumps(links), encoding="utf-8")
    records = []
    for day in original["study"]["dates"]:
        for kind in ("trading", "settlement"):
            identity = f"{kind}_{day.replace('-', '')}"
            content = snapshot(kind, day.replace("-", ""), original["study"]["contracts"])
            path = root / "seed" / f"{identity}.json"
            path.write_bytes(content)
            records.append(
                {"id": identity, "file": path.name, "sha256": hashlib.sha256(content).hexdigest()}
            )
    content = notice("cny2025")
    (root / "seed/holiday.html").write_bytes(content)
    records.append(
        {"id": "cny2025", "file": "holiday.html", "sha256": hashlib.sha256(content).hexdigest()}
    )
    (root / "input/seed_manifest.json").write_text(
        json.dumps({"records": records}), encoding="utf-8"
    )
    paths = {
        "calendar": "input/calendar.csv",
        "prior_exploration": "input/prior.csv",
        "seed": "seed",
        "seed_manifest": "input/seed_manifest.json",
        "previous_run": "input/previous.json",
        "discovered_links": "input/links.json",
        "raw": "raw",
        "output": "reports",
    }
    for key, value in paths.items():
        text = text.replace(f'{key} = "{original["paths"][key]}"', f'{key} = "{value}"')
        if key in original["input_hashes"]:
            text = text.replace(
                original["input_hashes"][key],
                hashlib.sha256((root / value).read_bytes()).hexdigest(),
            )
    config = config_dir / "review.toml"
    config.write_text(text, encoding="utf-8")
    plan = tomllib.loads(text)
    captures = []
    for item in plan["sources"]:
        if item["id"] == "cu_rules_202409":
            content = product()
        elif item["id"] == "rb_rules_202409":
            content = product("rb")
        elif item["id"] == "cu_rules_later_negative":
            content = product(later=True)
        elif item["id"] == "trading_rules_202409":
            content = trading_rule()
        elif item["id"] == "risk_rules_202409":
            content = "<p>发布日期：2024-09-03 自2024年10月23日起实施。交易保证金按照规定交易保证金比例中的最高值收取。</p>".encode()
        else:
            content = "<p>自2024年10月23日实施。</p>".encode()
        path = root / "raw" / f"{item['id']}.html"
        path.write_bytes(content)
        captures.append(
            {
                **item,
                "status": "http_ok",
                "file": path.name,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    (root / "raw/manifest.json").write_text(
        json.dumps(
            {
                "protocol_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "new_requests": 7,
                "records": captures,
            }
        ),
        encoding="utf-8",
    )
    return config


class ExecutionRuleReviewTests(unittest.TestCase):
    def test_later_rule_with_same_percentages_is_not_eligible_for_earlier_date(self):
        old, new = product_rule(product(), "cu"), product_rule(product(later=True), "cu")
        self.assertEqual(old["month_before_floor"], new["month_before_floor"])
        self.assertTrue(date_applicable(old, "2025-02-05"))
        self.assertFalse(date_applicable(new, "2025-02-05"))
        self.assertFalse(date_applicable(old, "2024-10-22"))
        self.assertEqual(old["multiplier_tonnes"], 5)

    def test_conflicting_table_or_effective_date_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicting"):
            product_rule(product() + "<p>交割月前第一月的第一个交易日起12%</p>".encode(), "cu")
        with self.assertRaisesRegex(ValueError, "effective date"):
            product_rule(product() + "<p>自2025年8月8日起实施</p>".encode(), "cu")

    def test_knowable_floor_is_maximum_not_sum_and_not_full_certification(self):
        result = known_margin_floor({"base_reference": 0.09, "near_delivery": 0.1})
        self.assertEqual(result["known_component_floor"], 0.1)
        self.assertIsNone(result["fully_resolved_rate"])
        self.assertEqual(result["binding_components"], "near_delivery")
        with self.assertRaises(ValueError):
            known_margin_floor({"unknown": float("nan")})

    def test_holiday_delays_first_month_session(self):
        dates = pd.bdate_range("2025-02-01", "2025-03-31")
        dates = dates[dates >= "2025-02-05"]
        calendar = pd.DataFrame({"date": dates})
        self.assertEqual(month_before_start("cu2503", calendar), pd.Timestamp("2025-02-05"))
        self.assertEqual(month_before_start("cu2504", calendar), pd.Timestamp("2025-03-03"))

    def test_auction_splits_submission_from_matching_without_invented_actual_open(self):
        row = scheduled_auction(notice("cny2025"), trading_rule())
        self.assertEqual(row["scheduled_submission_cutoff_at"], "2025-02-05T08:59:00+08:00")
        self.assertIsNone(row["actual_open_at"])
        self.assertFalse(row["actual_execution_clock_certified"])
        self.assertTrue(row["daily_open_can_be_later_first_trade"])
        changed = trading_rule().decode().replace("后1分钟", "后2分钟").encode()
        with self.assertRaisesRegex(ValueError, "durations disagree"):
            scheduled_auction(notice("cny2025"), changed)

    def test_complete_synthetic_review_and_tampered_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            review_plan(config)
            output = run_rule_review(config)
            summary = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["scenario_rows"], 48)
            self.assertEqual(summary["observed_reference_calibration_rows_in_each_scenario"], 3)
            self.assertEqual(summary["fully_resolved_margin_rates"], 0)
            self.assertEqual(summary["later_version_dates_excluded"], 4)
            self.assertEqual(summary["prediction_tests_added"], 0)
            spec = importlib.util.spec_from_file_location(
                "rule_review_checker", ROOT / "scripts/reconcile_execution_rule_review.py"
            )
            checker = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(checker)
            self.assertTrue(checker.reconcile(config, output, root / "good_audit"))
            path = output / "margin_scenarios.csv"
            original = path.read_bytes()
            wrong = pd.read_csv(path)
            wrong.loc[0, "known_component_floor"] += 0.01
            wrong.to_csv(path, index=False)
            self.assertFalse(checker.reconcile(config, output, root / "bad_audit"))
            path.write_bytes(original)
            (root / "raw/cu_rules_202409.html").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "differs from its capture"):
                run_rule_review(config)

    def test_independent_checker_detects_a_late_order_wrongly_accepted(self):
        modules = {}
        for name in ("demo_margin_phases", "reconcile_execution_rule_review"):
            spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules[name] = module
        with tempfile.TemporaryDirectory() as directory:
            output = modules["demo_margin_phases"].run_demo(Path(directory) / "demo")
            checker = modules["reconcile_execution_rule_review"]
            good = checker.Checks()
            checker.check_auction_files(output, good)
            self.assertEqual(good.differences, [])
            path = output / "auction_cutoff_checks.csv"
            wrong = pd.read_csv(path)
            wrong.loc[2, "rejected"] = False
            wrong.to_csv(path, index=False)
            bad = checker.Checks()
            checker.check_auction_files(output, bad)
            self.assertTrue(bad.differences)

    def test_protocol_cannot_drop_control_or_promote_source_certification(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            text = config.read_text(encoding="utf-8")
            config.write_text(
                text.replace(
                    "historical_first_vintage_certified = false",
                    "historical_first_vintage_certified = true",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "qualification boundaries"):
                review_plan(config)
            config.write_text(text, encoding="utf-8")
            (Path(directory) / "input/prior.csv").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input changed"):
                review_plan(config)
