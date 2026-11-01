"""Deliberate ledger corruption must be detected by the independent audit."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from test_market_benchmark import fixture

from commodity_factor_lab.execution import settle_portfolio
from commodity_factor_lab.market_benchmark import (
    attach_exposures,
    build_orders,
    prepare_marks,
    scenario_marks,
)

SOURCE = Path(__file__).resolve().parents[1] / "scripts/reconcile_market_benchmark.py"
SPEC = importlib.util.spec_from_file_location("market_benchmark_audit", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Audit, centered_rank, verify_book = MODULE.Audit, MODULE.centered_rank, MODULE.verify_book


class IndependentMarketAuditTests(unittest.TestCase):
    def test_tied_ranks(self):
        self.assertEqual(list(centered_rank([4, 4, 8, 1])), [0.0, 0.0, 1.5, -1.5])

    def test_independent_cash_detects_a_corrupted_fee(self):
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
        summary = SimpleNamespace(
            cost_bps=5.0,
            margin_rate=0.2,
            fill_scenario="assumed_full_open",
            final_equity=result.daily.equity.iloc[-1],
            fees=result.daily.fees.sum(),
            trade_legs=len(result.trades),
            rejected_product_orders=0,
            total_return=result.daily.equity.iloc[-1] / 100_000 - 1,
            annualized_return=(result.daily.equity.iloc[-1] / 100_000) ** (252 / len(clocks)) - 1,
            max_drawdown=0.0,
        )
        wealth = pd.Series([100_000.0, *result.daily.equity])
        summary.max_drawdown = (wealth / wealth.cummax() - 1).min()
        originals = {(r.date, r.contract): r for r in bars.itertuples()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("daily", "trades", "holdings", "rejected"):
                getattr(result, name).to_csv(root / f"{name}.csv", index=name == "daily")
            audit = Audit()
            verify_book(root, summary, originals, multipliers, 100_000, 252, audit)
            self.assertEqual(audit.differences, [])
            corrupted = pd.read_csv(root / "trades.csv")
            corrupted.loc[0, "fee"] += 100
            corrupted.to_csv(root / "trades.csv", index=False)
            audit = Audit()
            verify_book(root, summary, originals, multipliers, 100_000, 252, audit)
            self.assertTrue(any("/fee:" in x for x in audit.differences))


if __name__ == "__main__":
    unittest.main()
