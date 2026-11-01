"""Causality and state transitions of capacity-limited execution."""

import unittest

import numpy as np
import pandas as pd
from test_contract_foundation import example

from commodity_factor_lab.capacity import trailing_capacity
from commodity_factor_lab.execution import settle_portfolio


class CapacityTests(unittest.TestCase):
    def inputs(self):
        bars, calendar, orders = example()
        bars["max_trade_lots"] = 1
        bars["capacity_available_at"] = "2022-12-31T17:00:00+08:00"
        return bars, calendar, orders

    def test_past_window_includes_zeros_and_never_current_or_future_volume(self):
        bars, calendar, _ = example()
        bars.loc[bars.contract.eq("near"), "volume"] = [100, 0, 500, 9999]
        original = trailing_capacity(bars, calendar, 0.1, 2)
        row = original.loc[
            original.date.eq(calendar.date.iloc[2]) & original.contract.eq("near")
        ].iloc[0]
        self.assertEqual(row.max_trade_lots, 5)
        self.assertEqual(row.volume_window_end, calendar.date.iloc[1])
        bars.loc[bars.date.ge(calendar.date.iloc[2]), "volume"] *= 100
        changed = trailing_capacity(bars, calendar, 0.1, 2)
        pd.testing.assert_frame_equal(
            original.loc[original.date.le(calendar.date.iloc[2])],
            changed.loc[changed.date.le(calendar.date.iloc[2])],
        )

    def test_missing_calendar_session_is_not_replaced_by_older_observation(self):
        bars, calendar, _ = example()
        bars = bars.loc[~(bars.contract.eq("near") & bars.date.eq(calendar.date.iloc[1]))]
        result = trailing_capacity(bars, calendar, 0.1, 2)
        row = result.loc[result.date.eq(calendar.date.iloc[2]) & result.contract.eq("near")].iloc[0]
        self.assertEqual(row.max_trade_lots, 0)
        self.assertEqual(row.history_observations, 1)

    def test_invalid_limit_and_late_availability_are_rejected(self):
        for value in [-1, 0.5, np.nan]:
            bars, calendar, orders = self.inputs()
            bars["max_trade_lots"] = value
            with self.assertRaisesRegex(ValueError, "nonnegative integers"):
                settle_portfolio(bars, calendar, orders)
        bars, calendar, orders = self.inputs()
        bars["capacity_available_at"] = "2023-01-02T09:00:00+08:00"
        with self.assertRaisesRegex(ValueError, "submission deadline"):
            settle_portfolio(bars, calendar, orders)

    def test_atomic_policy_rejects_over_capacity_without_fee(self):
        bars, calendar, orders = self.inputs()
        orders["target_lots"] = 2
        result = settle_portfolio(bars, calendar, orders, liquidate=False)
        self.assertTrue(result.trades.empty)
        self.assertEqual(result.daily.fees.sum(), 0)
        self.assertTrue(result.rejected.reason.str.startswith("capacity_limit:").all())

    def test_partial_entry_retries_and_charges_only_actual_lots(self):
        bars, calendar, orders = self.inputs()
        orders["target_lots"] = 3
        result = settle_portfolio(bars, calendar, orders, liquidate=False, capacity_policy="staged")
        self.assertEqual(list(result.trades.delta_lots), [1, 1, 1])
        self.assertEqual(list(result.daily.contracts_held), [1, 1, 1, 1])
        self.assertEqual(result.holdings.lots.iloc[-1], 3)
        self.assertEqual(result.daily.fees.sum(), 3)
        self.assertEqual(list(result.capacity_events.remaining_lots), [2, 1, 0])

    def test_new_signal_cancels_unfilled_old_target(self):
        bars, calendar, orders = self.inputs()
        orders["target_lots"] = 3
        new = orders.iloc[0].to_dict() | {
            "date": calendar.date.iloc[1],
            "target_lots": 0,
            "decision_at": "2023-01-02T17:00:00+08:00",
        }
        result = settle_portfolio(
            bars,
            calendar,
            pd.concat([orders, pd.DataFrame([new])]),
            liquidate=False,
            capacity_policy="staged",
        )
        self.assertEqual(list(result.trades.delta_lots), [1, -1])
        self.assertEqual(result.daily.contracts_held.iloc[-1], 0)

    def test_roll_closes_old_before_opening_new_and_respects_each_leg_cap(self):
        bars, calendar, orders = self.inputs()
        orders["target_lots"] = 2
        bars.loc[
            bars.date.eq(calendar.date.iloc[0]) & bars.contract.eq("near"), "max_trade_lots"
        ] = 2
        new = orders.iloc[0].to_dict() | {
            "date": calendar.date.iloc[1],
            "contract": "far",
            "decision_at": "2023-01-02T17:00:00+08:00",
        }
        result = settle_portfolio(
            bars,
            calendar,
            pd.concat([orders, pd.DataFrame([new])]),
            liquidate=False,
            capacity_policy="staged",
        )
        day2 = result.holdings.loc[result.holdings.date.eq(calendar.date.iloc[1])]
        self.assertEqual(list(day2.contract), ["near"])
        self.assertEqual(list(day2.lots), [1])
        self.assertEqual(
            list(result.holdings.loc[result.holdings.date.eq(calendar.date.iloc[2]), "contract"]),
            ["far"],
        )
        self.assertEqual(result.holdings.lots.iloc[-1], 2)
        self.assertTrue(
            result.capacity_events.limit_reason.eq("deferred_until_old_contract_flat").any()
        )

    def test_blocked_old_exit_defers_new_and_later_retries(self):
        bars, calendar, orders = self.inputs()
        new = orders.iloc[0].to_dict() | {
            "date": calendar.date.iloc[1],
            "contract": "far",
            "decision_at": "2023-01-02T17:00:00+08:00",
        }
        bars.loc[bars.date.eq(calendar.date.iloc[1]) & bars.contract.eq("near"), "can_sell"] = False
        result = settle_portfolio(
            bars,
            calendar,
            pd.concat([orders, pd.DataFrame([new])]),
            liquidate=False,
            capacity_policy="staged",
        )
        self.assertEqual(
            list(result.holdings.loc[result.holdings.date.eq(calendar.date.iloc[1]), "contract"]),
            ["near"],
        )
        self.assertEqual(
            list(result.holdings.loc[result.holdings.date.eq(calendar.date.iloc[2]), "contract"]),
            ["far"],
        )

    def test_terminal_does_not_fake_a_complete_liquidation(self):
        bars, calendar, orders = self.inputs()
        orders["target_lots"] = 3
        result = settle_portfolio(bars, calendar, orders, capacity_policy="staged")
        self.assertTrue(result.daily.terminal_unliquidated.iloc[-1])
        self.assertEqual(result.holdings.lots.iloc[-1], 2)
        self.assertEqual(result.trades.delta_lots.iloc[-1], -1)

    def test_margin_rejection_does_not_report_or_charge_partial_fills(self):
        bars, calendar, orders = self.inputs()
        orders["target_lots"] = 3
        bars["max_trade_lots"] = 2
        result = settle_portfolio(
            bars,
            calendar,
            orders,
            initial_cash=1000,
            max_margin_fraction=0.15,
            liquidate=False,
            capacity_policy="staged",
        )
        self.assertTrue(result.trades.empty)
        self.assertEqual(result.daily.fees.sum(), 0)
        self.assertTrue(result.capacity_events.filled_delta_lots.eq(0).all())
        self.assertTrue(result.capacity_events.remaining_lots.eq(3).all())

    def test_forced_reduction_cancels_pending_entry_until_next_signal(self):
        bars, calendar, orders = self.inputs()
        orders["target_lots"] = 3
        calendar["settlement_at"] = [f"{d:%Y-%m-%d}T16:00:00+08:00" for d in calendar.date]
        bars["settlement_margin_rate"] = 0.3
        bars["settlement_spec_available_at"] = "2022-12-31T17:00:00+08:00"
        result = settle_portfolio(
            bars,
            calendar,
            orders,
            initial_cash=1000,
            max_margin_fraction=0.2,
            liquidate=False,
            capacity_policy="staged",
        )
        self.assertEqual(list(result.trades.delta_lots), [1, -1])
        self.assertTrue(result.daily.margin_call.iloc[0])
        self.assertEqual(result.daily.contracts_held.iloc[-1], 0)

    def test_reversal_consumes_both_closing_and_opening_quantity(self):
        bars, calendar, orders = self.inputs()
        new = orders.iloc[0].to_dict() | {
            "date": calendar.date.iloc[1],
            "target_lots": -1,
            "decision_at": "2023-01-02T17:00:00+08:00",
        }
        result = settle_portfolio(
            bars,
            calendar,
            pd.concat([orders, pd.DataFrame([new])]),
            liquidate=False,
            capacity_policy="staged",
        )
        self.assertEqual(list(result.trades.delta_lots), [1, -1, -1])
        self.assertEqual(result.trades.closed_lots.sum(), 1)
        self.assertEqual(result.trades.opened_lots.sum(), 2)
        self.assertEqual(result.daily.fees.sum(), 3)


if __name__ == "__main__":
    unittest.main()
