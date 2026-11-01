import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from commodity_factor_lab.engine import MissingHeldPrice
from commodity_factor_lab.portfolio import simulate
from commodity_factor_lab.protocol import StudyConfig
from commodity_factor_lab.study import build_signals, run_study, walk_forward_folds


def settings(**changes):
    return replace(StudyConfig(data=Path("data/input"), output=Path("reports/test")), **changes)


def panel():
    dates = pd.bdate_range("2023-01-02", periods=360)
    return pd.DataFrame(
        [
            {
                "date": date,
                "code": code,
                "open": 100.0 + i,
                "close": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "volume": 10000,
            }
            for code in ("a", "b", "c", "cs", "m", "y")
            for i, date in enumerate(dates)
        ]
    )


class ProtocolTests(unittest.TestCase):
    def test_training_labels_finish_strictly_before_test_entry(self):
        folds = walk_forward_folds(pd.bdate_range("2020-01-01", periods=800), settings())
        self.assertTrue((folds["train_last_label_end"] < folds["test_entry_start"]).all())
        self.assertTrue(
            (
                folds["test_entry_start"].iloc[1:].to_numpy()
                > folds["test_entry_end"].iloc[:-1].to_numpy()
            ).all()
        )

    def test_future_prices_cannot_change_past_eligibility_or_signals(self):
        data = panel()
        signals, _, eligible = build_signals(data, settings())
        cutoff = data.date.max().to_pydatetime() - timedelta(days=20)
        changed = data.copy()
        for col in ("open", "high", "low", "close"):
            changed.loc[changed.date >= cutoff, col] *= 1.6
        after, _, after_eligible = build_signals(changed, settings())
        pd.testing.assert_frame_equal(
            eligible.loc[eligible.index < cutoff], after_eligible.loc[after_eligible.index < cutoff]
        )
        for name in signals:
            pd.testing.assert_frame_equal(
                signals[name].loc[signals[name].index < cutoff],
                after[name].loc[after[name].index < cutoff],
            )

    def test_long_history_and_common_signal_support_are_required(self):
        signals, _, eligible = build_signals(panel(), settings())
        self.assertFalse(eligible.iloc[:251].any().any())
        for frame in signals.values():
            pd.testing.assert_frame_equal(frame.notna(), eligible)

    def test_missing_price_resets_the_causal_history_requirement(self):
        data = panel()
        code_mask = data.code.eq("a")
        gap = data.loc[code_mask].iloc[250].date
        data.loc[code_mask & data.date.eq(gap), "close"] = np.nan
        _, _, eligible = build_signals(data, settings())
        self.assertFalse(eligible.loc[gap:, "a"].any())

    def test_primary_cost_cannot_be_picked_outside_declared_scenarios(self):
        with self.assertRaises(ValueError):
            settings(primary_cost_bps=7)


class ScheduledPortfolioTests(unittest.TestCase):
    def test_unscaled_per_asset_positions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "gross exposure"):
            simulate(
                pd.DataFrame([[1.0, -1.0]]), pd.DataFrame([[0.0, 0.0]]), pd.Series([True]), 0.001
            )

    def test_non_rebalance_day_keeps_units_and_has_no_artificial_trades(self):
        dates = pd.bdate_range("2023-01-02", periods=3)
        targets = pd.DataFrame([[0.5, -0.5]] * 3, index=dates)
        returns = pd.DataFrame([[0.1, 0], [0.1, 0], [0.1, 0]], index=dates)
        result = simulate(
            targets, returns, pd.Series([True, False, False], index=dates), 0.001, liquidate=False
        )
        self.assertEqual(result.daily.iloc[1]["turnover"], 0)
        self.assertAlmostEqual(result.weights.iloc[1, 0], 0.55 / 1.05)
        self.assertGreater(result.daily.iloc[1]["gross_return"], 0.05)

    def test_initial_entry_and_terminal_exit_are_both_charged(self):
        targets = pd.DataFrame([[0.5, -0.5], [0.5, -0.5]])
        result = simulate(targets, targets * 0, pd.Series([True, False]), 0.001)
        self.assertGreater(result.daily.iloc[0]["entry_cost"], 0)
        self.assertAlmostEqual(result.daily.iloc[-1]["terminal_cost"], 0.001)
        self.assertAlmostEqual((1 + result.daily.net_return).prod(), 0.999 / 1.001)

    def test_missing_mark_is_detected_for_drifted_holdings(self):
        targets = pd.DataFrame([[0.5, -0.5], [0.0, 0.0]])
        returns = pd.DataFrame([[0.0, 0.0], [np.nan, 0.0]])
        with self.assertRaises(MissingHeldPrice):
            simulate(targets, returns, pd.Series([True, False]), 0.001)

    def test_missing_calendar_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            simulate(pd.DataFrame([[1], [1]]), pd.DataFrame([[0], [0]]), pd.Series([True]), 0.001)


class ProvenanceTests(unittest.TestCase):
    def test_deleting_a_downloaded_product_does_not_silently_change_the_universe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "trend").mkdir()
            source = root / "trend/Sina_a_daily.csv"
            panel().loc[lambda x: x.code.eq("a")].drop(columns="code").to_csv(source, index=False)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            records = [
                {"status": "ok", "normalized_file": name, "normalized_sha256": digest}
                for name in ("trend/Sina_a_daily.csv", "trend/Sina_b_daily.csv")
            ]
            (root / "snapshot.json").write_text(json.dumps({"records": records}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "recorded normalized hashes"):
                run_study(settings(data=root, output=root.parent / "unused-study-output"))
