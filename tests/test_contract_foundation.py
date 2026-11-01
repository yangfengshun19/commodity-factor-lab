"""Checkpoint tests for the new contract research components (not end-to-end)."""

import json
import unittest

import numpy as np
import pandas as pd

from commodity_factor_lab.curve_factors import CurveRules, contract_curves
from commodity_factor_lab.execution import settle_portfolio
from commodity_factor_lab.fundamentals import (
    asof_fundamentals,
    normalized_position_flow,
    seasonal_inventory,
)
from commodity_factor_lab.official import parse_contract_metadata


def example():
    dates = pd.bdate_range("2023-01-02", periods=4)
    calendar = pd.DataFrame(
        {
            "date": dates,
            "open_at": [f"{d:%Y-%m-%d}T09:00:00+08:00" for d in dates],
            "signal_cutoff": [f"{d:%Y-%m-%d}T17:00:00+08:00" for d in dates],
        }
    )
    bars = pd.DataFrame(
        [
            {
                "date": date,
                "code": "rb",
                "contract": name,
                "open": price,
                "settle": price,
                "volume": 10000,
                "open_interest": 10000,
                "multiplier": 10,
                "margin_rate": 0.1,
                "fee_rate": 0,
                "fee_per_lot": 1,
                "close_fee_rate": 0,
                "close_fee_per_lot": 1,
                "can_buy": True,
                "can_sell": True,
                "spec_available_at": "2022-12-01T16:00:00+08:00",
                "expiry": expiry,
                "available_at": f"{date:%Y-%m-%d}T16:00:00+08:00",
            }
            for date in dates
            for name, price, expiry in [
                ("near", 100, "2023-04-20"),
                ("far", 200, "2023-07-20"),
                ("third", 300, "2023-10-20"),
            ]
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "date": dates[0],
                "contract": "near",
                "target_lots": 1,
                "decision_at": "2023-01-01T17:00:00+08:00",
            }
        ]
    )
    return bars, calendar, orders


class SettlementTests(unittest.TestCase):
    def test_auction_deadline_rejects_a_decision_before_continuous_open(self):
        bars, calendar, orders = example()
        calendar["settlement_at"] = [f"{d:%Y-%m-%d}T16:00:00+08:00" for d in calendar.date]
        calendar["order_cutoff_at"] = [f"{d:%Y-%m-%d}T08:59:00+08:00" for d in calendar.date]
        orders.loc[0, "decision_at"] = "2023-01-02T08:59:30+08:00"
        with self.assertRaisesRegex(ValueError, "order submission cutoff"):
            settle_portfolio(bars, calendar, orders)
        orders.loc[0, "decision_at"] = "2023-01-02T08:59:00+08:00"
        with self.assertRaisesRegex(ValueError, "order submission cutoff"):
            settle_portfolio(bars, calendar, orders)
        orders.loc[0, "decision_at"] = "2023-01-02T08:58:59+08:00"
        self.assertTrue(settle_portfolio(bars, calendar, orders).rejected.empty)

    def test_order_cutoff_cannot_precede_the_prior_settlement_or_follow_open(self):
        bars, calendar, orders = example()
        calendar["settlement_at"] = [f"{d:%Y-%m-%d}T16:00:00+08:00" for d in calendar.date]
        calendar["order_cutoff_at"] = [f"{d:%Y-%m-%d}T08:59:00+08:00" for d in calendar.date]
        calendar.loc[1, "order_cutoff_at"] = "2023-01-02T15:59:00+08:00"
        with self.assertRaisesRegex(ValueError, "previous settlement"):
            settle_portfolio(bars, calendar, orders)
        calendar.loc[1, "order_cutoff_at"] = "2023-01-03T09:00:01+08:00"
        with self.assertRaisesRegex(ValueError, "must not follow"):
            settle_portfolio(bars, calendar, orders)

    def test_auction_cutoff_requires_settlement_clock_for_forced_order_causality(self):
        bars, calendar, orders = example()
        calendar["order_cutoff_at"] = [f"{d:%Y-%m-%d}T08:59:00+08:00" for d in calendar.date]
        with self.assertRaisesRegex(ValueError, "explicit settlement clocks"):
            settle_portfolio(bars, calendar, orders)

    def test_settlement_clock_alone_can_accompany_unchanged_margin(self):
        bars, calendar, orders = example()
        calendar["settlement_at"] = [f"{d:%Y-%m-%d}T16:00:00+08:00" for d in calendar.date]
        result = settle_portfolio(bars, calendar, orders, initial_cash=1000)
        self.assertEqual(result.daily.iloc[0].reserved_margin, 100)

    def test_settlement_margin_increase_does_not_reject_an_earlier_opening(self):
        bars, calendar, orders = example()
        calendar["settlement_at"] = [f"{d:%Y-%m-%d}T16:00:00+08:00" for d in calendar.date]
        bars["settlement_margin_rate"] = 0.3
        bars["settlement_spec_available_at"] = "2023-01-02T12:00:00+08:00"
        bars.loc[bars.date.gt(calendar.iloc[0].date), "margin_rate"] = 0.3
        result = settle_portfolio(
            bars, calendar, orders, initial_cash=1000, max_margin_fraction=0.2, liquidate=False
        )
        self.assertTrue(result.rejected.empty)
        self.assertEqual(result.daily.iloc[0].opening_reserved_margin, 100)
        self.assertEqual(result.daily.iloc[0].reserved_margin, 300)
        self.assertEqual(result.daily.iloc[0].equity, 999)
        self.assertTrue(result.daily.iloc[0].margin_call)
        self.assertEqual(result.trades.iloc[1].reason, "margin_liquidation")
        self.assertEqual(result.daily.iloc[1].reserved_margin, 0)
        self.assertTrue(result.daily.reconciliation_error.eq(0).all())

    def test_incomplete_or_future_settlement_parameter_is_rejected(self):
        bars, calendar, orders = example()
        bars["settlement_margin_rate"] = 0.3
        with self.assertRaisesRegex(ValueError, "Phased margin requires"):
            settle_portfolio(bars, calendar, orders)
        bars["settlement_spec_available_at"] = "2023-01-02T17:00:00+08:00"
        calendar["settlement_at"] = [f"{d:%Y-%m-%d}T16:00:00+08:00" for d in calendar.date]
        with self.assertRaisesRegex(ValueError, "unavailable at settlement"):
            settle_portfolio(bars, calendar, orders)
        bars["settlement_spec_available_at"] = "2022-12-01T16:00:00+08:00"
        bars.loc[0, "settlement_margin_rate"] = 0
        with self.assertRaisesRegex(ValueError, "Invalid settlement margin"):
            settle_portfolio(bars, calendar, orders)

    def test_settlement_clock_cannot_cross_next_night_opening(self):
        bars, calendar, orders = example()
        bars["settlement_margin_rate"] = 0.3
        bars["settlement_spec_available_at"] = "2022-12-01T16:00:00+08:00"
        calendar["settlement_at"] = [f"{d:%Y-%m-%d}T16:00:00+08:00" for d in calendar.date]
        calendar.loc[0, "settlement_at"] = "2023-01-03T10:00:00+08:00"
        with self.assertRaisesRegex(ValueError, "precede the next opening"):
            settle_portfolio(bars, calendar, orders)

    def test_unknown_expiry_cannot_silently_bypass_delivery_controls(self):
        bars, calendar, orders = example()
        bars.loc[bars.contract.eq("near"), "expiry"] = None
        with self.assertRaisesRegex(ValueError, "expiries"):
            settle_portfolio(bars, calendar, orders)

    def test_roll_gap_is_not_profit_and_both_trade_legs_pay_fees(self):
        bars, calendar, orders = example()
        roll = pd.DataFrame(
            [
                {
                    "date": calendar.iloc[1].date,
                    "contract": "far",
                    "target_lots": 1,
                    "decision_at": "2023-01-02T17:00:00+08:00",
                }
            ]
        )
        result = settle_portfolio(bars, calendar, pd.concat([orders, roll]), initial_cash=1000)
        self.assertEqual(result.daily.iloc[-1].equity, 996)
        self.assertEqual(len(result.trades), 4)
        self.assertTrue(result.daily.reconciliation_error.eq(0).all())
        self.assertEqual(result.daily.iloc[-1].reserved_margin, 0)

    def test_margin_is_reserved_and_not_subtracted_twice_from_equity(self):
        bars, calendar, orders = example()
        result = settle_portfolio(bars, calendar, orders, initial_cash=1000, liquidate=False)
        self.assertEqual(result.daily.iloc[0].equity, 999)
        self.assertEqual(result.daily.iloc[0].reserved_margin, 100)
        self.assertEqual(result.daily.iloc[0].available_cash, 899)

    def test_overnight_and_intraday_cash_pnl_reconcile(self):
        bars, calendar, orders = example()
        mask = bars.contract.eq("near")
        bars.loc[mask, "open"] = [100, 108, 110, 110]
        bars.loc[mask, "settle"] = [105, 110, 110, 110]
        result = settle_portfolio(bars, calendar, orders, initial_cash=1000)
        self.assertEqual(result.daily.iloc[0].equity, 1049)
        self.assertEqual(result.daily.iloc[1].overnight_pnl, 30)
        self.assertEqual(result.daily.iloc[1].intraday_pnl, 20)
        self.assertEqual(result.daily.iloc[-1].equity, 1098)

    def test_blocked_roll_does_not_open_second_contract_or_charge_failed_order(self):
        bars, calendar, orders = example()
        bars.loc[bars.date.eq(calendar.iloc[1].date) & bars.contract.eq("far"), "can_buy"] = False
        roll = pd.DataFrame(
            [
                {
                    "date": calendar.iloc[1].date,
                    "contract": "far",
                    "target_lots": 1,
                    "decision_at": "2023-01-02T17:00:00+08:00",
                }
            ]
        )
        result = settle_portfolio(bars, calendar, pd.concat([orders, roll]), initial_cash=1000)
        self.assertEqual(result.daily.iloc[-1].equity, 998)
        self.assertEqual(len(result.rejected), 1)
        self.assertFalse(result.holdings.contract.eq("far").any())

    def test_future_specifications_and_late_night_orders_are_rejected(self):
        bars, calendar, orders = example()
        bars["spec_available_at"] = "2024-01-01T16:00:00+08:00"
        with self.assertRaisesRegex(ValueError, "specifications"):
            settle_portfolio(bars, calendar, orders)
        bars, calendar, orders = example()
        orders.loc[0, "decision_at"] = "2023-01-02T10:00:00+08:00"
        with self.assertRaisesRegex(ValueError, "precede"):
            settle_portfolio(bars, calendar, orders)

    def test_missing_held_price_is_not_replaced_with_a_neighbour_contract(self):
        bars, calendar, orders = example()
        bars = bars.loc[~(bars.date.eq(calendar.iloc[1].date) & bars.contract.eq("near"))]
        with self.assertRaisesRegex(ValueError, "Missing held"):
            settle_portfolio(bars, calendar, orders)

    def test_insufficient_margin_rejects_entry(self):
        bars, calendar, orders = example()
        orders["target_lots"] = 10
        result = settle_portfolio(bars, calendar, orders, initial_cash=1000)
        self.assertEqual(result.daily.iloc[-1].equity, 1000)
        self.assertEqual(result.rejected.iloc[0].reason, "insufficient_margin_budget")


class CurveTests(unittest.TestCase):
    def test_curve_rank_changes_never_create_momentum(self):
        bars, calendar, _ = example()
        metadata = bars.drop_duplicates("contract")[["contract", "expiry"]].copy()
        metadata["metadata_available_at"] = "2022-12-01T16:00:00+08:00"
        bars.loc[bars.date.ge(calendar.iloc[1].date) & bars.contract.eq("near"), "volume"] = 0
        result = contract_curves(
            bars, metadata, calendar, CurveRules(momentum_sessions=2, basis_momentum_sessions=2)
        )
        np.testing.assert_allclose(result.rank1_return.dropna(), 0)
        self.assertEqual(result.iloc[1].front_contract, "far")
        self.assertEqual(result.iloc[-1].momentum_126d, 0)

    def test_unknown_expiry_does_not_get_guessed(self):
        bars, calendar, _ = example()
        metadata = bars.drop_duplicates("contract")[["contract", "expiry"]].copy()
        metadata["metadata_available_at"] = "2024-01-01T16:00:00+08:00"
        result = contract_curves(bars, metadata, calendar)
        self.assertTrue(result.carry.isna().all())

    def test_future_prices_cannot_change_earlier_curve_features(self):
        bars, calendar, _ = example()
        metadata = bars.drop_duplicates("contract")[["contract", "expiry"]].copy()
        metadata["metadata_available_at"] = "2022-12-01T16:00:00+08:00"
        before = contract_curves(bars, metadata, calendar)
        bars.loc[bars.date.ge(calendar.iloc[2].date), "settle"] *= 1.5
        after = contract_curves(bars, metadata, calendar)
        pd.testing.assert_frame_equal(before.iloc[:2], after.iloc[:2])


class FundamentalTests(unittest.TestCase):
    def observations(self):
        return pd.DataFrame(
            [
                {
                    "code": "rb",
                    "period_date": f"{year}-01-02",
                    "available_at": f"{year}-01-02T16:00:00+08:00",
                    "value": value,
                    "unit": "tonnes",
                    "source": "test",
                }
                for year, value in [(2020, 10), (2021, 20), (2022, 30), (2023, 40)]
            ]
        )

    def test_future_revisions_cannot_rewrite_past_inventory(self):
        data = self.observations()
        calendar = pd.DataFrame(
            {"date": ["2023-01-02"], "signal_cutoff": ["2023-01-02T17:00:00+08:00"]}
        )
        before = seasonal_inventory(data, calendar, min_observations=3)
        revision = data.iloc[[0]].copy()
        revision["available_at"] = "2024-01-01T16:00:00+08:00"
        revision["value"] = 999
        after = seasonal_inventory(pd.concat([data, revision]), calendar, min_observations=3)
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(before.iloc[0].inventory_scarcity, -0.5)

    def test_units_cannot_change_silently(self):
        data = self.observations()
        data.loc[0, "unit"] = "lots"
        calendar = pd.DataFrame(
            {"date": ["2023-01-02"], "signal_cutoff": ["2023-01-02T17:00:00+08:00"]}
        )
        with self.assertRaisesRegex(ValueError, "units"):
            asof_fundamentals(data, calendar)

    def test_position_flow_requires_complete_history_and_uses_level_oi(self):
        positions = pd.DataFrame({"rb": [0.0, 1.0, 2.0, 3.0, 4.0]})
        oi = pd.DataFrame({"rb": [100.0] * 5})
        self.assertAlmostEqual(normalized_position_flow(positions, oi, 2).iloc[-1, 0], 0.02)
        positions.iloc[2, 0] = np.nan
        self.assertTrue(np.isnan(normalized_position_flow(positions, oi, 2).iloc[-1, 0]))

    def test_official_metadata_preserves_real_holiday_adjusted_expiry(self):
        raw = json.dumps(
            {
                "ContractBaseInfo": [
                    {
                        "INSTRUMENTID": "cu2402",
                        "EXPIREDATE": "20240219",
                        "OPENDATE": "20230216",
                        "UPDATE_DATE": "2024-01-30 19:51:09",
                    }
                ]
            }
        ).encode()
        data = parse_contract_metadata(raw, "official-test")
        self.assertEqual(str(data.iloc[0].expiry.date()), "2024-02-19")
        self.assertEqual(str(data.iloc[0].metadata_available_at), "2024-01-30 11:51:09+00:00")
