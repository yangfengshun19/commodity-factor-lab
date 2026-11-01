"""Small, hand-calculable cases for timing, capital normalization and costs."""

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from commodity_factor_lab.config import ResearchConfig
from commodity_factor_lab.engine import (
    evaluate_factor,
    performance,
    portfolio_ledger,
    quantile_groups,
    signal_weights,
    summarize,
)


def config(**changes):
    base = ResearchConfig(
        data=Path("data/input"),
        output=Path("reports/test"),
        families=("trend",),
        start="2023-01-01",
        end="2023-12-31",
        test_start="2023-01-09",
        publication_lag_sessions=0,
        fee_bps=0,
        slippage_bps=0,
    )
    return replace(base, **changes)


def panel():
    dates = pd.bdate_range("2023-01-02", periods=9)
    return pd.DataFrame(
        [
            {
                "date": date,
                "code": str(asset),
                "open": 100 * 1.1 ** max(0, day - 1) if asset == 4 else 100.0,
                "t_mom": float(asset),
            }
            for day, date in enumerate(dates)
            for asset in range(5)
        ]
    )


class TimingTests(unittest.TestCase):
    def test_signal_at_close_earns_only_following_open_to_open(self):
        result = evaluate_factor(panel(), "t_mom", config())
        self.assertEqual(result.ledger.index[0], pd.Timestamp("2023-01-04"))
        self.assertEqual(result.ledger.iloc[0]["signal_date"], pd.Timestamp("2023-01-02"))
        self.assertEqual(result.ledger.iloc[0]["entry_date"], pd.Timestamp("2023-01-03"))
        np.testing.assert_allclose(result.ledger["gross_return"], 0.05)

    def test_publication_lag_waits_one_more_session(self):
        result = evaluate_factor(panel(), "t_mom", config(publication_lag_sessions=1))
        self.assertEqual(result.ledger.index[0], pd.Timestamp("2023-01-05"))

    def test_direction_is_explicit_and_not_chosen_from_returns(self):
        result = evaluate_factor(panel(), "t_mom", config(directions={"t_mom": -1}))
        np.testing.assert_allclose(result.ledger["gross_return"], -0.05)

    def test_future_price_changes_cannot_change_signal_weights(self):
        data = panel()
        before = evaluate_factor(data, "t_mom", config())
        data.loc[data["date"] >= "2023-01-10", "open"] *= 1.7
        after = evaluate_factor(data, "t_mom", config())
        pd.testing.assert_frame_equal(before.weights, after.weights)

    def test_future_signal_changes_cannot_change_earlier_returns(self):
        data = panel()
        before = evaluate_factor(data, "t_mom", config())
        data.loc[data["date"] >= "2023-01-09", "t_mom"] *= -1
        after = evaluate_factor(data, "t_mom", config())
        pd.testing.assert_frame_equal(
            before.ledger.loc[:"2023-01-10"], after.ledger.loc[:"2023-01-10"]
        )

    def test_missing_held_mark_invalidates_backtest(self):
        data = panel()
        data.loc[(data["code"] == "4") & (data["date"] == "2023-01-05"), "open"] = np.nan
        result = evaluate_factor(data, "t_mom", config())
        self.assertTrue(result.ledger.empty)
        self.assertIn("Missing opening price", result.error)
        self.assertEqual(summarize(result, config())[0]["status"], "missing_held_price")

    def test_train_ic_purges_labels_ending_in_test(self):
        data = panel()
        dates = data["date"].drop_duplicates().sort_values()
        result = evaluate_factor(data, "t_mom", config())
        self.assertTrue(pd.isna(result.ic.loc[dates.iloc[3]]))
        self.assertTrue(pd.isna(result.ic.loc[dates.iloc[4]]))

    def test_test_portfolio_is_assigned_by_entry_date(self):
        result = evaluate_factor(panel(), "t_mom", config())
        records = {r["period"]: r for r in summarize(result, config())}
        self.assertEqual(records["test"]["start"], "2023-01-10")
        self.assertEqual(
            records["full"]["days"], records["train"]["days"] + records["test"]["days"]
        )


class PortfolioTests(unittest.TestCase):
    def test_gross_exposure_does_not_grow_with_number_of_names(self):
        values = pd.DataFrame([range(5), range(5)])
        labels = quantile_groups(values, 5, 5)
        weights = signal_weights(labels, 5)
        np.testing.assert_allclose(weights.abs().sum(axis=1), 1)
        np.testing.assert_allclose(weights.sum(axis=1), 0)

    def test_ties_and_small_cross_sections_abstain(self):
        labels = quantile_groups(
            pd.DataFrame([[1, 1, 1, 1, 1], [1, 2, np.nan, np.nan, np.nan]]), 5, 5
        )
        self.assertTrue(labels.isna().all().all())

    def test_equal_factor_values_are_never_arbitrarily_split(self):
        labels = quantile_groups(pd.DataFrame([[1, 2, 3, 4, 5, 5]]), 5, 5)
        self.assertEqual(labels.iloc[0, 4], labels.iloc[0, 5])

    def test_entry_cost_is_charged_even_on_first_holding_day(self):
        weights = pd.DataFrame([[0.5, -0.5]])
        returns = pd.DataFrame([[0.1, 0.0]])
        ledger = portfolio_ledger(weights, returns, 0.001)
        self.assertAlmostEqual(ledger.iloc[0]["cost"], 0.001 / 1.001)
        self.assertAlmostEqual(ledger.iloc[0]["net_return"], 1.05 / 1.001 - 1)

    def test_drift_requires_rebalance_even_when_target_is_unchanged(self):
        ledger = portfolio_ledger(
            pd.DataFrame([[0.5, -0.5], [0.5, -0.5]]), pd.DataFrame([[0.0, 0.1], [0.0, 0.0]]), 0
        )
        self.assertAlmostEqual(
            ledger.iloc[1]["turnover"], abs(0.5 - 0.5 / 0.95) + abs(-0.5 + 0.55 / 0.95)
        )

    def test_exit_cost_and_initial_drawdown(self):
        ledger = portfolio_ledger(
            pd.DataFrame([[0.5, -0.5], [0, 0]]), pd.DataFrame([[0, 0], [0, 0]]), 0.001
        )
        self.assertGreater(ledger.iloc[1]["cost"], 0)
        self.assertAlmostEqual(performance(pd.Series([-0.1]))["max_drawdown"], -0.1)

    def test_short_history_is_visible(self):
        record = summarize(evaluate_factor(panel(), "t_mom", config()), config())[0]
        self.assertEqual(record["status"], "short_history")


if __name__ == "__main__":
    unittest.main()
