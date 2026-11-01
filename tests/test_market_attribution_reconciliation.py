"""Independent cashflow path and intentional contribution-corruption checks."""

import importlib.util
import unittest
from pathlib import Path

from test_market_attribution import accounting_fixture

from commodity_factor_lab.market_attribution import attribute_book

SOURCE = Path(__file__).resolve().parents[1] / "scripts/reconcile_market_attribution.py"
SPEC = importlib.util.spec_from_file_location("market_attribution_audit", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class IndependentAttributionTests(unittest.TestCase):
    def test_independent_trade_cashflows_match_hand_calculated_roll(self):
        bars, daily, trades, holdings = accounting_fixture()
        parts = MODULE.cash_parts(bars, daily, trades, holdings, ["a", "b"], {"a": 10.0, "b": 2.0})
        self.assertAlmostEqual(parts.net_pnl.sum(), 271.494)
        self.assertAlmostEqual(parts.loc[parts.code.eq("a"), "gross_pnl"].sum(), 250.0)
        self.assertAlmostEqual(parts.loc[parts.code.eq("b"), "gross_pnl"].sum(), 26.0)

    def test_deliberate_wrong_contribution_is_detected(self):
        args = accounting_fixture()
        expected = MODULE.cash_parts(*args, ["a", "b"], {"a": 10.0, "b": 2.0})
        actual, _ = attribute_book(*args, ["a", "b"], {"a": 10.0, "b": 2.0}, 5.0, "manual")
        checks = MODULE.Checks()
        MODULE.compare_frame(
            checks, actual, expected, ["date", "code"], ["net_pnl", "gross_pnl"], "fixture"
        )
        self.assertEqual(checks.differences, [])
        actual.loc[0, "net_pnl"] += 100
        checks = MODULE.Checks()
        MODULE.compare_frame(
            checks, actual, expected, ["date", "code"], ["net_pnl", "gross_pnl"], "fixture"
        )
        self.assertEqual(len(checks.differences), 1)


if __name__ == "__main__":
    unittest.main()
