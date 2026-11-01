import math
import unittest
from datetime import datetime, timezone

import pandas as pd

from commodity_factor_lab.reconciliation import (
    Checks,
    clock,
    compounded,
    reference_season,
    reference_selection,
    workbook_identity,
)


class IndependentReferenceTests(unittest.TestCase):
    def test_option_block_must_not_inherit_previous_futures_identity(self):
        self.assertEqual(workbook_identity(math.nan, "cu2109"), "cu2109")
        self.assertIsNone(workbook_identity("cu2010C33500", "cu2109"))
        self.assertIsNone(workbook_identity(math.nan, None))
        self.assertIsNone(workbook_identity("小计", "cu2109"))

    def test_checks_detect_corruption_and_handle_unavailable_values(self):
        checks = Checks()
        checks.add("carry", "2024-01-01", "cu", 0.02, 0.03)
        checks.add("missing", "2024-01-01", "cu", math.nan, math.nan)
        checks.add("presence", "2024-01-01", "cu", True, True)
        self.assertEqual([r["status"] for r in checks.rows], ["difference", "match", "match"])

    def test_compounding_keeps_full_windows_and_missing_sessions(self):
        self.assertAlmostEqual(compounded([0.1, -0.1], 2), -0.01)
        self.assertTrue(math.isnan(compounded([0.1], 2)))
        self.assertTrue(math.isnan(compounded([0.1, math.nan], 2)))

    def test_date_only_is_not_a_publication_clock(self):
        self.assertIsNone(clock("20240102"))
        self.assertEqual(clock("20240102 16:00:00").hour, 8)

    def test_reference_selection_ignores_future_expiry_information_and_roll_gap(self):
        dates = pd.bdate_range("2023-01-02", periods=3)
        bars = pd.DataFrame(
            [
                {
                    "date": d,
                    "code": "cu",
                    "contract": c,
                    "settle": p,
                    "volume": 1000,
                    "open_interest": 1000,
                }
                for d in dates
                for c, p in [("near", 100.0), ("far", 200.0), ("future", 50.0)]
            ]
        )
        metadata = [
            {"contract": c, "expiry": pd.Timestamp(e).date(), "published": clock(t)}
            for c, e, t in [
                ("near", "2023-03-04", "2022-12-01 16:00:00"),
                ("far", "2023-06-01", "2022-12-01 16:00:00"),
                ("future", "2023-04-01", "2023-01-05 16:00:00"),
            ]
        ]
        sessions = pd.DataFrame(
            {"date": dates, "signal_cutoff": [clock(f"{d:%Y-%m-%d} 20:30:00") for d in dates]}
        )
        rules = {"min_days": 60, "max_days": 540, "min_volume": 100, "min_open_interest": 1000}
        selected, rows = reference_selection(bars, metadata, sessions, rules)
        self.assertEqual(selected[(dates[0], "cu")][0]["contract"], "near")
        self.assertEqual(selected[(dates[-1], "cu")][0]["contract"], "far")
        self.assertEqual(rows[-1]["rank1_return"], 0)
        self.assertTrue(all("future" not in [x["contract"] for x in v] for v in selected.values()))

    def test_seasonal_reference_excludes_recent_december_and_future_publications(self):
        rows = [
            {
                "date": pd.Timestamp(f"{y}-01-02").date(),
                "published": clock(f"{y}-01-02 16:00:00"),
                "value": v,
            }
            for y, v in [(2020, 10), (2021, 20), (2022, 30), (2023, 40)]
        ]
        when = pd.Timestamp("2023-01-02")
        before = reference_season(rows, when, datetime(2023, 1, 2, 12, tzinfo=timezone.utc), 3, 3)
        rows.append(
            {
                "date": pd.Timestamp("2022-12-30").date(),
                "published": clock("2022-12-30 16:00:00"),
                "value": 0,
            }
        )
        rows.append(
            {
                "date": pd.Timestamp("2020-01-02").date(),
                "published": clock("2023-01-03 16:00:00"),
                "value": 1000,
            }
        )
        after = reference_season(rows, when, datetime(2023, 1, 2, 12, tzinfo=timezone.utc), 3, 3)
        self.assertEqual(before["expected"], -0.5)
        self.assertEqual(before, after)
