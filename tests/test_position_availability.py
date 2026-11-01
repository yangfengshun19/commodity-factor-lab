import unittest

import numpy as np
import pandas as pd

from commodity_factor_lab.position_availability import replay_position_availability


class PositionAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2024-01-02", periods=25)
        self.calendar = pd.DataFrame(
            {
                "date": self.dates,
                "signal_cutoff": [
                    date.tz_localize("UTC") + pd.Timedelta(750, unit="m") for date in self.dates
                ],
            }
        )
        self.qualified = pd.DataFrame(
            [
                {
                    "date": date,
                    "contract": contract,
                    "status": "complete_top20",
                    "long_top20": base + i * 20,
                    "short_top20": 1000,
                    "open_interest": 10000,
                    "available_at": date.tz_localize("UTC") + pd.Timedelta(12, unit="h"),
                }
                for i, date in enumerate(self.dates)
                for contract, base in (("cu2404", 2000), ("cu2405", 5000))
            ]
        )
        self.curves = pd.DataFrame(
            {"date": self.dates, "code": "cu", "front_contract": ["cu2404"] * 20 + ["cu2405"] * 5}
        )

    def test_late_history_recovers_only_after_its_actual_file_clock(self):
        mask = self.qualified.date.eq(self.dates[5]) & self.qualified.contract.eq("cu2405")
        self.qualified.loc[mask, "status"] = "published_after_cutoff"
        self.qualified.loc[mask, "available_at"] = self.calendar.iloc[22].signal_cutoff
        result = replay_position_availability(self.qualified, self.curves, self.calendar)
        self.assertTrue(pd.isna(result.iloc[20].normalized_position_flow))
        self.assertTrue(result.iloc[20].structural_full_window)
        self.assertEqual(result.iloc[20].status, "window_version_not_yet_known")
        self.assertAlmostEqual(result.iloc[22].normalized_position_flow, 0.04)
        self.assertAlmostEqual(result.iloc[24].normalized_position_flow, 0.04)
        self.assertEqual(
            result.iloc[22].latest_window_available_at, self.calendar.iloc[22].signal_cutoff
        )

    def test_a_future_current_record_does_not_backfill_today(self):
        mask = self.qualified.date.eq(self.dates[24]) & self.qualified.contract.eq("cu2405")
        self.qualified.loc[mask, "available_at"] = self.calendar.iloc[
            24
        ].signal_cutoff + pd.Timedelta(1, unit="ns")
        self.qualified.loc[mask, "status"] = "published_after_cutoff"
        result = replay_position_availability(self.qualified, self.curves, self.calendar)
        self.assertTrue(pd.isna(result.iloc[-1].normalized_position_flow))
        self.assertEqual(result.iloc[-1].known_window_points, 20)
        self.qualified.loc[mask, "available_at"] = self.calendar.iloc[24].signal_cutoff
        result = replay_position_availability(self.qualified, self.curves, self.calendar)
        self.assertAlmostEqual(result.iloc[-1].normalized_position_flow, 0.04)

    def test_roll_looks_back_at_the_same_previously_unselected_contract(self):
        result = replay_position_availability(self.qualified, self.curves, self.calendar)
        self.assertAlmostEqual(result.iloc[20].normalized_position_flow, 0.04)
        self.assertEqual(result.iloc[20].contract, "cu2405")
        self.assertEqual(result.iloc[19].status, "warmup")

    def test_missing_middle_or_unknown_clock_cannot_be_recovered(self):
        mask = self.qualified.date.eq(self.dates[5]) & self.qualified.contract.eq("cu2405")
        for mode in ("missing", "unknown_clock", "invalid_rank", "overflow"):
            source = self.qualified.copy()
            if mode == "missing":
                source = source.loc[~mask]
            elif mode == "unknown_clock":
                source.loc[mask, "available_at"] = pd.NaT
            elif mode == "invalid_rank":
                source.loc[mask, "status"] = "incomplete_or_invalid_ranks"
            else:
                source.loc[mask, "long_top20"] = 10001
            with self.subTest(mode=mode):
                result = replay_position_availability(source, self.curves, self.calendar)
                self.assertTrue(pd.isna(result.iloc[-1].normalized_position_flow))
                self.assertFalse(result.iloc[-1].structural_full_window)

    def test_calendar_and_identity_duplicates_are_rejected(self):
        with self.assertRaises(ValueError):
            replay_position_availability(
                pd.concat([self.qualified, self.qualified.iloc[:1]]), self.curves, self.calendar
            )
        bad = self.calendar.copy()
        bad.loc[24, "date"] = bad.loc[23, "date"]
        with self.assertRaises(ValueError):
            replay_position_availability(self.qualified, self.curves, bad)

    def test_ignored_clock_coverage_flag_is_not_a_predictive_value(self):
        source = self.qualified.copy()
        source["status"] = "published_after_cutoff"
        source["available_at"] = self.calendar.iloc[-1].signal_cutoff + pd.Timedelta(1, unit="d")
        result = replay_position_availability(source, self.curves, self.calendar)
        self.assertTrue(result.iloc[20:].structural_full_window.all())
        self.assertTrue(np.isnan(result.normalized_position_flow).all())
