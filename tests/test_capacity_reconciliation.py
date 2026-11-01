"""An independent auditor must detect corrupted capacity and fee records."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from test_contract_foundation import example

from commodity_factor_lab.capacity_study import compare_original
from commodity_factor_lab.execution import settle_portfolio

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reconcile_capacity.py"
SPEC = importlib.util.spec_from_file_location("capacity_auditor", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class CapacityReconciliationTests(unittest.TestCase):
    def test_independent_cash_and_past_volume_detect_corruption(self):
        bars, calendar, orders = example()
        bars["volume"] = 100
        bars["max_trade_lots"] = 1
        bars["capacity_available_at"] = "2022-12-31T17:00:00+08:00"
        bars["fee_per_lot"] = bars["close_fee_per_lot"] = 0
        bars["fee_rate"] = bars["close_fee_rate"] = 0.0005
        orders["target_lots"] = 3
        result = settle_portfolio(bars, calendar, orders, liquidate=False, capacity_policy="staged")
        # Two actual earlier sessions provide the first execution's history.
        dates = list(pd.bdate_range(end=calendar.date.iloc[0], periods=3)[:-1]) + list(
            calendar.date
        )
        earlier = [bars.loc[bars.date.eq(calendar.date.iloc[0])].assign(date=d) for d in dates[:2]]
        raw = pd.concat([*earlier, bars], ignore_index=True)
        # With no terminal liquidation requested, the last saved flag must
        # identify the residual holdings for this independent terminal check.
        result.daily.loc[result.daily.index[-1], "terminal_unliquidated"] = True
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("daily", "trades", "holdings", "capacity_events"):
                getattr(result, name).to_csv(folder / f"{name}.csv", index=name == "daily")
            args = (folder, raw, dates, {"rb": 10}, 1_000_000, 0.01, 2, 5, 0.1)
            good = AUDIT.verify_book(*args)
            self.assertEqual(good["differences"], [])
            trades = pd.read_csv(folder / "trades.csv")
            trades.loc[0, "fee"] += 10
            trades.to_csv(folder / "trades.csv", index=False)
            bad = AUDIT.verify_book(*args)
            self.assertTrue(any("/fee" in x for x in bad["differences"]))
            trades.loc[0, "fee"] -= 10
            trades.to_csv(folder / "trades.csv", index=False)
            events = pd.read_csv(folder / "capacity_events.csv")
            events.loc[0, "max_trade_lots"] = 999
            events.to_csv(folder / "capacity_events.csv", index=False)
            bad = AUDIT.verify_book(*args)
            self.assertTrue(any("/budget" in x for x in bad["differences"]))

    def test_baseline_comparison_detects_changed_cash(self):
        bars, calendar, orders = example()
        result = settle_portfolio(bars, calendar, orders)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("daily", "trades", "holdings", "rejected"):
                getattr(result, name).to_csv(folder / f"{name}.csv", index=name == "daily")
            compare_original(result, folder)
            daily = pd.read_csv(folder / "daily.csv")
            daily.loc[0, "equity"] += 1
            daily.to_csv(folder / "daily.csv", index=False)
            with self.assertRaises(AssertionError):
                compare_original(result, folder)


if __name__ == "__main__":
    unittest.main()
