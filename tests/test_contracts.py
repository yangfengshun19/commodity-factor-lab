import unittest

import numpy as np
import pandas as pd

from commodity_factor_lab.contracts import same_contract_panel


def contract_inputs():
    dates = pd.bdate_range("2023-01-02", periods=5)
    calendar = pd.DataFrame(
        {
            "date": dates,
            "open_at": [f"{d:%Y-%m-%d}T09:00:00+08:00" for d in dates],
            "signal_cutoff": [f"{d:%Y-%m-%d}T16:00:00+08:00" for d in dates],
        }
    )
    bars = pd.DataFrame(
        [
            {
                "date": d,
                "code": "rb",
                "contract": contract,
                "open": price,
                "volume": 100,
                "open_interest": (100 if i == 0 else 300) if contract == "far" else 200,
                "multiplier": 10,
                "expiry": "2023-06-20",
                "available_at": f"{d:%Y-%m-%d}T15:30:00+08:00",
            }
            for i, d in enumerate(dates)
            for contract, price in [("near", 100.0), ("far", 200.0)]
        ]
    )
    return bars, calendar


class ContractTests(unittest.TestCase):
    def test_roll_price_difference_never_becomes_an_investment_return(self):
        bars, calendar = contract_inputs()
        result = same_contract_panel(bars, calendar)
        np.testing.assert_allclose(result["return"], 0)
        self.assertEqual(result.iloc[0]["contract"], "near")
        self.assertEqual(result.iloc[1]["contract"], "far")
        self.assertTrue(result.iloc[1]["rolled"])
        self.assertEqual(result.iloc[1]["entry_trade_legs"], 2)

    def test_future_open_interest_cannot_change_past_selection(self):
        bars, calendar = contract_inputs()
        before = same_contract_panel(bars, calendar)
        bars.loc[bars.date >= calendar.iloc[2].date, "open_interest"] *= 1000
        after = same_contract_panel(bars, calendar)
        pd.testing.assert_series_equal(before.iloc[0], after.iloc[0])

    def test_delayed_publication_is_not_available_to_the_signal(self):
        bars, calendar = contract_inputs()
        bars.loc[bars.date.eq(calendar.iloc[0].date), "available_at"] = "2023-01-02T17:00:00+08:00"
        result = same_contract_panel(bars, calendar)
        self.assertEqual(result.iloc[0]["status"], "no_known_eligible_contract")

    def test_multiplier_is_used_in_cash_pnl(self):
        bars, calendar = contract_inputs()
        bars.loc[bars.date.eq(calendar.iloc[2].date) & bars.contract.eq("near"), "open"] = 105
        first = same_contract_panel(bars, calendar).iloc[0]
        self.assertAlmostEqual(first["return"], 0.05)
        self.assertAlmostEqual(first["one_contract_pnl"], 50)

    def test_naive_publication_timestamp_is_rejected(self):
        bars, calendar = contract_inputs()
        bars["available_at"] = "2023-01-02 15:30:00"
        with self.assertRaises(ValueError):
            same_contract_panel(bars, calendar)

    def test_negative_futures_price_uses_cash_pnl_without_fake_percentage(self):
        bars, calendar = contract_inputs()
        bars.loc[bars.date.eq(calendar.iloc[1].date) & bars.contract.eq("near"), "open"] = -10
        bars.loc[bars.date.eq(calendar.iloc[2].date) & bars.contract.eq("near"), "open"] = -5
        first = same_contract_panel(bars, calendar).iloc[0]
        self.assertEqual(first["one_contract_pnl"], 50)
        self.assertTrue(np.isnan(first["return"]))
        self.assertEqual(first["status"], "cash_pnl_only_nonpositive_price")

    def test_missing_same_contract_mark_is_never_filled_by_other_contract(self):
        bars, calendar = contract_inputs()
        bars = bars.loc[~(bars.date.eq(calendar.iloc[2].date) & bars.contract.eq("near"))]
        first = same_contract_panel(bars, calendar).iloc[0]
        self.assertEqual(first["status"], "missing_same_contract_mark")
        self.assertTrue(np.isnan(first["return"]))
