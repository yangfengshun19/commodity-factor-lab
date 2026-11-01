import hashlib
import importlib.util
import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from commodity_factor_lab.execution_readiness import (
    GATES,
    admission,
    fee_candidate,
    parse_notice,
    parse_parameter_snapshot,
    run_execution_readiness,
)
from commodity_factor_lab.execution_sources import capture_execution_sources, execution_plan

ROOT = Path(__file__).resolve().parents[1]


def notice(identity):
    if identity == "rb2410_april":
        text = "发布日期：2024-03-28 自2024年4月1日交易（即3月29日晚夜盘）起：螺纹钢期货RB2410合约交易手续费调整为成交金额的万分之一，日内平今仓交易手续费调整为成交金额的万分之一。"
    elif identity == "rb_offcycle_may":
        text = "发布日期：2024-05-29 自2024年5月31日交易（即5月30日晚夜盘）起：螺纹钢期货除1月、5月、10月合约以外其他月合约的交易手续费调整为成交金额的万分之零点二，日内平今仓交易手续费调整为成交金额的万分之零点二。当合约进入交割月前第二月的第一个交易日起，交易手续费恢复到成交金额的万分之一。"
    else:
        text = "发布日期：2025-01-22 自2025年1月24日收盘结算时起：铜、铝、锌、铅期货合约的涨跌停板幅度调整为10%，套保交易保证金比例调整为11%，投机交易保证金比例调整为12%；螺纹钢、热轧卷板、不锈钢期货合约的涨跌停板幅度调整为8%，套保交易保证金比例调整为9%，投机交易保证金比例调整为10%。2025年1月27日晚上不进行夜盘交易。2025年2月5日08:55-09:00集合竞价，当晚恢复夜盘交易。自第一个未出现单边市的交易日收盘结算时，恢复至原有水平。"
    return ("<html><script>ignore</script><p>" + text + "</p></html>").encode()


def snapshot(kind, day, contracts, update=None):
    rows = []
    for contract in contracts:
        margin = (0.12 if contract.startswith("cu") else 0.1) if kind == "settlement" else 0.07
        rate = (
            0.0001
            if contract == "rb2410" or (contract == "rb2408" and day == "20240603")
            else 0.00002
        )
        rows.append(
            {
                "INSTRUMENTID": contract.upper(),
                "TRADINGDAY": day,
                "SPEC_LONGMARGINRATIO": str(margin),
                "SPEC_SHORTMARGINRATIO": str(margin),
                "TRADEFEEUNIT": "0",
                "TRADEFEERATION": str(rate),
            }
        )
    return json.dumps(
        {
            "report_date": day,
            "update_date": update or f"{day} 16:00:00",
            "Settlement" if kind == "settlement" else "ContractDailyTradeArgument": rows,
        }
    ).encode()


def fixture(root):
    (root / "configs").mkdir()
    (root / "inputs").mkdir()
    (root / "seed").mkdir()
    (root / "src/commodity_factor_lab").mkdir(parents=True)
    for name in ("execution_readiness.py", "execution_sources.py", "execution.py"):
        (root / "src/commodity_factor_lab" / name).write_bytes(
            (ROOT / "src/commodity_factor_lab" / name).read_bytes()
        )
    calendar = pd.DataFrame({"date": pd.bdate_range("2024-01-01", "2025-12-31")})
    calendar.to_csv(root / "inputs/calendar.csv", index=False)
    pd.DataFrame({"analysis": list(range(15))}).to_csv(root / "inputs/prior.csv", index=False)
    text = (ROOT / "configs/execution_readiness.toml").read_text(encoding="utf-8")
    raw = tomllib.loads(text)
    for key, path in {
        "calendar": "inputs/calendar.csv",
        "prior_exploration": "inputs/prior.csv",
        "seed": "seed",
        "raw": "raw",
        "output": "reports",
    }.items():
        text = text.replace(f'{key} = "{raw["paths"][key]}"', f'{key} = "{path}"')
        if key in raw["input_hashes"]:
            text = text.replace(
                raw["input_hashes"][key], hashlib.sha256((root / path).read_bytes()).hexdigest()
            )
    for source in raw["sources"]:
        if "seed_file" in source:
            content = snapshot(source["kind"], source["date"], [], "20260205 17:03:09")
            (root / "seed" / source["seed_file"]).write_bytes(content)
            text = text.replace(source["seed_sha256"], hashlib.sha256(content).hexdigest())
    config = root / "configs/execution.toml"
    config.write_text(text, encoding="utf-8")
    return config, calendar


class ExecutionReadinessTests(unittest.TestCase):
    def test_dated_fee_units_and_effective_night_are_parsed(self):
        parsed = parse_notice(notice("rb_offcycle_may"), "rb_offcycle_may")
        self.assertEqual(parsed["reduced_fee_rate"], 0.00002)
        self.assertEqual(parsed["restored_fee_rate"], 0.0001)
        self.assertEqual(parsed["effective_trading_date"], "2024-05-31")
        self.assertEqual(parsed["effective_night_natural_date"], "2024-05-30")
        april = parse_notice(notice("rb2410_april"), "rb2410_april")
        self.assertEqual(april["effective_night_natural_date"], "2024-03-29")
        self.assertEqual(april["effective_trading_date"], "2024-04-01")

    def test_fee_month_exception_boundary_and_pre_effective_negative_control(self):
        notices = {
            key: parse_notice(notice(key), key) for key in ("rb2410_april", "rb_offcycle_may")
        }
        calendar = pd.DataFrame({"date": pd.bdate_range("2024-01-01", "2024-12-31")})
        self.assertTrue(
            np.isnan(
                fee_candidate("2024-05-30", "rb2408", notices, calendar)["notice_expected_rate"]
            )
        )
        self.assertEqual(
            fee_candidate("2024-05-31", "rb2408", notices, calendar)["notice_expected_rate"],
            0.00002,
        )
        self.assertEqual(
            fee_candidate("2024-06-03", "rb2408", notices, calendar)["notice_expected_rate"], 0.0001
        )
        self.assertEqual(
            fee_candidate("2024-06-03", "rb2409", notices, calendar)["notice_expected_rate"],
            0.00002,
        )
        self.assertEqual(
            fee_candidate("2024-05-31", "rb2410", notices, calendar)["notice_expected_rate"], 0.0001
        )
        with self.assertRaises(ValueError):
            fee_candidate("2024-05-31", "rb2413", notices, calendar)

    def test_changed_fee_and_missing_conditional_recovery_are_rejected(self):
        changed = notice("rb_offcycle_may").decode().replace("零点二", "零点三").encode()
        with self.assertRaises(ValueError):
            parse_notice(changed, "rb_offcycle_may")
        changed = notice("cny2025").decode().replace("未出现单边市", "无条件").encode()
        with self.assertRaises(ValueError):
            parse_notice(changed, "cny2025")
        parsed = parse_notice(notice("cny2025"), "cny2025")
        self.assertTrue(parsed["restoration_conditional"])
        self.assertEqual(parsed["rates"]["cu"]["spec_margin"], 0.12)
        self.assertEqual(parsed["rates"]["rb"]["hedge_margin"], 0.09)

    def test_parameter_report_date_duplicate_and_update_clock_controls(self):
        content = snapshot("settlement", "20250124", ["cu2503"], "20260205 17:03:09")
        rows, audit = parse_parameter_snapshot(content, "settlement", "20250124", ["cu2503"])
        self.assertTrue(audit["source_update_after_report_date"])
        self.assertFalse(rows[0]["snapshot_is_first_vintage"])
        with self.assertRaisesRegex(ValueError, "report date"):
            parse_parameter_snapshot(content, "settlement", "20250127", ["cu2503"])
        data = json.loads(content)
        data["Settlement"] *= 2
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_parameter_snapshot(
                json.dumps(data).encode(), "settlement", "20250124", ["cu2503"]
            )

    def test_source_agreement_alone_cannot_admit_real_execution(self):
        evidence = dict.fromkeys(GATES, True)
        evidence["directional_fill_evidence"] = False
        eligible, blockers = admission(evidence)
        self.assertFalse(eligible)
        self.assertIn("directional_fill_evidence", blockers)
        evidence["directional_fill_evidence"] = "True"
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            admission(evidence)

    def test_end_to_end_synthetic_assay_keeps_gates_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = fixture(root)
            plan, paths = execution_plan(config)
            paths["raw"].mkdir()
            records = []
            for source in plan["sources"]:
                if "seed_file" in source:
                    content = (root / "seed" / source["seed_file"]).read_bytes()
                elif source["kind"] == "notice":
                    content = notice(source["id"])
                elif source["kind"] == "current_hours":
                    content = b"<p>21:00 current only</p>"
                else:
                    wanted = (
                        []
                        if "seed_file" in source
                        else (
                            plan["study"]["margin_contracts"]
                            if source["date"].startswith("2025")
                            else plan["study"]["fee_contracts"]
                        )
                    )
                    content = snapshot(source["kind"], source["date"], wanted)
                path = paths["raw"] / (source["id"] + ".fixture")
                path.write_bytes(content)
                records.append(
                    {
                        **source,
                        "file": path.name,
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "status": "seed_reused" if "seed_file" in source else "http_ok",
                    }
                )
            (paths["raw"] / "manifest.json").write_text(
                json.dumps(
                    {
                        "protocol_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                        "records": records,
                        "new_requests": 14,
                    }
                ),
                encoding="utf-8",
            )
            output = run_execution_readiness(config)
            summary = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["normalized_parameter_rows"], 22)
            self.assertEqual(summary["fee_snapshot_matches"], 6)
            self.assertEqual(summary["prediction_tests_added"], 0)
            self.assertFalse(summary["real_execution_eligible"])
            self.assertEqual(
                pd.read_csv(output / "session_evidence.csv").actual_open_at.notna().sum(), 0
            )
            spec = importlib.util.spec_from_file_location(
                "execution_checker", ROOT / "scripts/reconcile_execution_readiness.py"
            )
            checker = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(checker)
            self.assertTrue(checker.reconcile(config, output, root / "good_audit"))
            fee_path = output / "fee_cases.csv"
            original_fees = fee_path.read_bytes()
            wrong = pd.read_csv(fee_path)
            wrong.loc[4, "notice_expected_rate"] = 0.01
            wrong.to_csv(fee_path, index=False)
            self.assertFalse(checker.reconcile(config, output, root / "bad_audit"))
            fee_path.write_bytes(original_fees)
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "differs from its receipt"):
                run_execution_readiness(config)

    def test_margin_phase_demo_and_independent_cash_check(self):
        modules = {}
        for name in ("demo_margin_phases", "reconcile_execution_readiness"):
            spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules[name] = module
        with tempfile.TemporaryDirectory() as directory:
            target = modules["demo_margin_phases"].run_demo(Path(directory) / "demo")
            checker = modules["reconcile_execution_readiness"]
            good = checker.Checks()
            checker.check_margin_demo(target, good)
            self.assertEqual(good.differences, [])
            daily = pd.read_csv(target / "daily.csv")
            daily.loc[0, "equity"] -= 200
            daily.to_csv(target / "daily.csv", index=False)
            bad = checker.Checks()
            checker.check_margin_demo(target, bad)
            self.assertTrue(bad.differences)

    def test_capture_429_stops_and_remains_stopped_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = fixture(Path(directory))
            response = Mock(status=429)
            response.getheader.return_value = None
            response.read.return_value = b"rate limited"
            connection = Mock()
            connection.getresponse.return_value = response
            with (
                patch(
                    "commodity_factor_lab.execution_sources.http.client.HTTPSConnection",
                    return_value=connection,
                ),
                patch("commodity_factor_lab.execution_sources.time.sleep"),
            ):
                receipt = capture_execution_sources(config)
                self.assertEqual(receipt["new_requests"], 1)
                self.assertEqual(len(receipt["records"]), 1)
                with self.assertRaisesRegex(ValueError, "stop remains active"):
                    capture_execution_sources(config)

    def test_transport_failures_stop_after_three_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = fixture(Path(directory))
            connection = Mock()
            connection.request.side_effect = OSError("synthetic unavailable")
            with (
                patch(
                    "commodity_factor_lab.execution_sources.http.client.HTTPSConnection",
                    return_value=connection,
                ),
                patch("commodity_factor_lab.execution_sources.time.sleep"),
            ):
                receipt = capture_execution_sources(config)
                self.assertEqual(receipt["new_requests"], 3)
                self.assertTrue(all(row["status"] == "failed" for row in receipt["records"]))

    def test_frozen_input_change_and_real_claim_relaxation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = fixture(root)
            text = config.read_text(encoding="utf-8")
            config.write_text(
                text.replace("actual_fill_certified = false", "actual_fill_certified = true"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "claim gates"):
                execution_plan(config)
            config.write_text(text, encoding="utf-8")
            (root / "inputs/calendar.csv").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input changed"):
                execution_plan(config)
