"""Causality, common samples and independent cash identities for market scenarios."""

import unittest

import numpy as np
import pandas as pd

from commodity_factor_lab.execution import settle_portfolio
from commodity_factor_lab.market_benchmark import (
    BOOKS,
    append_exploration,
    attach_exposures,
    build_orders,
    prepare_marks,
    scenario_marks,
    summarize,
    weights,
)


def fixture(assets=8, periods=12):
    dates = pd.bdate_range("2023-01-02", periods=periods)
    calendar = pd.DataFrame(
        {"date": dates, "signal_cutoff": [f"{d:%Y-%m-%d}T20:30:00+08:00" for d in dates]}
    )
    curves, bars = [], []
    codes = [f"p{i}" for i in range(assets)]
    for number, day in enumerate(dates):
        for i, code in enumerate(codes):
            contract = f"{code}{'a' if number < 5 else 'b'}"
            curves.append(
                {
                    "date": day,
                    "code": code,
                    "front_contract": contract,
                    "front_expiry": "2024-06-17",
                    "reference_settle": 100 + i,
                    "carry": 7 - i,
                    "momentum_126d": i,
                }
            )
            for suffix in ("a", "b"):
                price = 100 + i + number
                bars.append(
                    {
                        "date": day,
                        "code": code,
                        "contract": code + suffix,
                        "open": price,
                        "high": price + 2,
                        "low": price - 2,
                        "close": price + 1,
                        "settle": price + 1,
                    }
                )
    study = {
        "start": str(dates[1].date()),
        "end": str(dates[-1].date()),
        "codes": codes,
        "min_assets": 8,
        "rebalance_sessions": 5,
        "initial_cash": 100_000,
        "gross_notional_on_initial_cash": 1,
    }
    return pd.DataFrame(curves), calendar, pd.DataFrame(bars), study, dict.fromkeys(codes, 1.0)


class MarketBenchmarkTests(unittest.TestCase):
    def test_excluded_product_exits_above_threshold_and_can_reenter(self):
        curves, calendar, bars, study, multipliers = fixture(9, 17)
        curves.loc[curves.date.eq(calendar.date.iloc[5]) & curves.code.eq("p0"), "carry"] = np.nan
        orders, audit, clocks = build_orders(curves, calendar, study, multipliers)
        marks = prepare_marks(bars, curves, clocks, orders, multipliers)
        result = settle_portfolio(
            scenario_marks(marks, 5, 0.2, "assumed_full_open"),
            clocks,
            orders["equal_weight_long"],
            100_000,
        )
        self.assertFalse(
            (result.holdings.date.eq(calendar.date.iloc[6]) & result.holdings.code.eq("p0")).any()
        )
        self.assertTrue(
            (result.holdings.date.eq(calendar.date.iloc[11]) & result.holdings.code.eq("p0")).any()
        )
        self.assertTrue(audit.status.eq("excluded_product_zero_target").any())

    def test_exclusion_closes_old_contract_after_rejected_roll(self):
        curves, calendar, bars, study, multipliers = fixture(9, 17)
        curves.loc[curves.date.eq(calendar.date.iloc[10]) & curves.code.eq("p0"), "carry"] = np.nan
        orders, _, clocks = build_orders(curves, calendar, study, multipliers)
        marks = scenario_marks(
            prepare_marks(bars, curves, clocks, orders, multipliers), 5, 0.2, "assumed_full_open"
        )
        marks.loc[marks.date.eq(calendar.date.iloc[6]) & marks.contract.eq("p0b"), "can_buy"] = (
            False
        )
        result = settle_portfolio(marks, clocks, orders["equal_weight_long"], 100_000)
        self.assertTrue(
            (
                result.holdings.date.eq(calendar.date.iloc[6]) & result.holdings.contract.eq("p0a")
            ).any()
        )
        self.assertFalse(
            (result.holdings.date.eq(calendar.date.iloc[11]) & result.holdings.code.eq("p0")).any()
        )

    def test_blocked_exclusion_exit_remains_visible(self):
        curves, calendar, bars, study, multipliers = fixture(9)
        curves.loc[curves.date.eq(calendar.date.iloc[5]) & curves.code.eq("p0"), "carry"] = np.nan
        orders, _, clocks = build_orders(curves, calendar, study, multipliers)
        marks = scenario_marks(
            prepare_marks(bars, curves, clocks, orders, multipliers), 5, 0.2, "assumed_full_open"
        )
        marks.loc[marks.date.eq(calendar.date.iloc[6]) & marks.contract.eq("p0a"), "can_sell"] = (
            False
        )
        result = settle_portfolio(marks, clocks, orders["equal_weight_long"], 100_000)
        self.assertTrue(result.rejected.reason.str.contains("directional_fill_blocked:p0a").any())
        self.assertTrue(
            (result.holdings.date.eq(calendar.date.iloc[6]) & result.holdings.code.eq("p0")).any()
        )

    def test_missing_open_is_preserved_and_required_position_still_fails(self):
        curves, calendar, bars, study, multipliers = fixture()
        orders, _, clocks = build_orders(curves, calendar, study, multipliers)
        mask = bars.date.eq(calendar.date.iloc[2]) & bars.contract.eq("p0a")
        bars.loc[mask, "open"] = np.nan
        marks = prepare_marks(bars, curves, clocks, orders, multipliers)
        self.assertTrue(marks.open.isna().any())
        with self.assertRaisesRegex(ValueError, "Missing held/ordered"):
            settle_portfolio(
                scenario_marks(marks, 5, 0.2, "assumed_full_open"),
                clocks,
                orders["equal_weight_long"],
                100_000,
            )

    def test_expiry_revision_is_applied_only_after_its_signal_date(self):
        curves, calendar, bars, study, multipliers = fixture()
        mask = curves.front_contract.eq("p0a") & curves.date.ge(calendar.date.iloc[3])
        curves.loc[mask, "front_expiry"] = "2024-06-24"
        orders, _, clocks = build_orders(curves, calendar, study, multipliers)
        marks = prepare_marks(bars, curves, clocks, orders, multipliers)
        old = marks.loc[marks.contract.eq("p0a") & marks.date.eq(calendar.date.iloc[3])].iloc[0]
        new = marks.loc[marks.contract.eq("p0a") & marks.date.eq(calendar.date.iloc[4])].iloc[0]
        self.assertEqual(old.expiry, "2024-06-17")
        self.assertEqual(new.expiry, "2024-06-24")

    def test_ranks_and_complete_offset_keep_cash(self):
        group = pd.DataFrame({"momentum_126d": [1, 2, 3], "carry": [3, 2, 1]})
        np.testing.assert_allclose(weights(group, "momentum"), [-0.5, 0, 0.5])
        np.testing.assert_allclose(weights(group, "momentum_carry"), [0, 0, 0])
        np.testing.assert_allclose(weights(group, "equal_weight_long"), [1 / 3] * 3)

    def test_next_session_signal_and_previous_settlement_size(self):
        curves, calendar, _, study, multipliers = fixture()
        orders, audit, clocks = build_orders(curves, calendar, study, multipliers)
        first = audit.loc[audit.book.eq("momentum")].iloc[0]
        self.assertEqual(first.signal_date, calendar.date.iloc[0])
        self.assertEqual(first.date, calendar.date.iloc[1])
        self.assertEqual(first.target_lots, int(np.trunc(first.weight * 100_000 / 100)))
        self.assertTrue((clocks.open_at < clocks.settlement_at).all())
        self.assertTrue(orders["momentum"].date.isin(calendar.date.iloc[[1, 6]]).all())
        self.assertFalse(orders["momentum"].date.eq(calendar.date.iloc[-1]).any())

    def test_future_perturbation_does_not_change_earlier_targets(self):
        curves, calendar, _, study, multipliers = fixture()
        original, _, _ = build_orders(curves, calendar, study, multipliers)
        curves.loc[curves.date.ge(calendar.date.iloc[6]), ["carry", "momentum_126d"]] *= -1000
        changed, _, _ = build_orders(curves, calendar, study, multipliers)
        for book in BOOKS:
            pd.testing.assert_frame_equal(original[book], changed[book])

    def test_common_universe_failure_flattens_all_books(self):
        curves, calendar, _, study, multipliers = fixture()
        curves.loc[curves.date.eq(calendar.date.iloc[5]) & curves.code.eq("p0"), "carry"] = np.nan
        orders, audit, _ = build_orders(curves, calendar, study, multipliers)
        for book in BOOKS:
            self.assertTrue(
                orders[book]
                .loc[orders[book].date.eq(calendar.date.iloc[6]), "target_lots"]
                .eq(0)
                .all()
            )
        self.assertTrue(audit.status.eq("insufficient_common_universe").any())

    def test_roll_fees_terminal_and_independent_cash_sum(self):
        curves, calendar, bars, study, multipliers = fixture()
        orders, _, clocks = build_orders(curves, calendar, study, multipliers)
        marks = prepare_marks(bars, curves, clocks, orders, multipliers)
        result = settle_portfolio(
            scenario_marks(marks, 5, 0.2, "assumed_full_open"),
            clocks,
            orders["equal_weight_long"],
            100_000,
        )
        attach_exposures(result, multipliers)
        trades = result.trades
        self.assertAlmostEqual(result.daily.fees.sum(), trades.notional.sum() * 0.0005)
        self.assertTrue(
            trades.loc[trades.date.eq(calendar.date.iloc[6]), "closed_lots"].gt(0).any()
        )
        self.assertTrue(
            trades.loc[trades.date.eq(calendar.date.iloc[6]), "opened_lots"].gt(0).any()
        )
        self.assertEqual(result.daily.contracts_held.iloc[-1], 0)
        # Every cashflow can be reconstructed from buys/sells and daily settlement;
        # terminal-flat cash equals the sum of trade cashflows minus explicit cost.
        expected = 100_000 - (trades.delta_lots * trades.fill_price).sum() - trades.fee.sum()
        self.assertAlmostEqual(result.daily.equity.iloc[-1], expected)
        self.assertAlmostEqual(summarize(result, 100_000)["net_pnl"], expected - 100_000)

    def test_flat_ohlc_is_an_explicit_stress_not_a_selection_filter(self):
        curves, calendar, bars, study, multipliers = fixture()
        orders, _, clocks = build_orders(curves, calendar, study, multipliers)
        marks = prepare_marks(bars, curves, clocks, orders, multipliers)
        mask = marks.date.eq(calendar.date.iloc[1]) & marks.code.eq("p0")
        for col in ("high", "low", "close"):
            marks.loc[mask, col] = marks.loc[mask, "open"]
        stress = scenario_marks(marks, 5, 0.2, "block_flat_ohlc")
        self.assertFalse(stress.loc[mask, "can_buy"].any())
        self.assertTrue(scenario_marks(marks, 5, 0.2, "assumed_full_open").can_buy.all())
        result = settle_portfolio(stress, clocks, orders["equal_weight_long"], 100_000)
        self.assertTrue(result.rejected.reason.str.contains("directional_fill_blocked").any())

    def test_missing_held_mark_fails_instead_of_filling(self):
        curves, calendar, bars, study, multipliers = fixture()
        orders, _, clocks = build_orders(curves, calendar, study, multipliers)
        marks = prepare_marks(bars, curves, clocks, orders, multipliers)
        marks = marks.loc[~(marks.date.eq(calendar.date.iloc[2]) & marks.contract.eq("p0a"))]
        with self.assertRaisesRegex(ValueError, "Missing held/ordered"):
            settle_portfolio(
                scenario_marks(marks, 5, 0.2, "assumed_full_open"),
                clocks,
                orders["equal_weight_long"],
                100_000,
            )

    def test_nonpositive_recorded_open_cannot_be_filled(self):
        curves, calendar, bars, study, multipliers = fixture()
        orders, _, clocks = build_orders(curves, calendar, study, multipliers)
        bars.loc[bars.date.eq(calendar.date.iloc[2]), "open"] = 0
        with self.assertRaisesRegex(ValueError, "Missing/nonpositive"):
            prepare_marks(bars, curves, clocks, orders, multipliers)

    def test_exploration_registry_preserves_old_fields(self):
        prior = pd.DataFrame(
            {
                "analysis": ["old"] * 15,
                "hac_p": [0.001] + [np.nan] * 14,
                "family_size": [15] * 15,
                "claim_eligible": [False] * 15,
            }
        )
        new = append_exploration(prior)
        pd.testing.assert_frame_equal(new.iloc[:15][prior.columns], prior, check_dtype=False)
        self.assertEqual(len(new), 16)
        self.assertTrue(new.family_16_size.eq(16).all())
        self.assertGreater(new.family_16_by_q.iloc[0], 0.001)


if __name__ == "__main__":
    unittest.main()
