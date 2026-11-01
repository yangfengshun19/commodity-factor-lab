"""Shared eligibility, calendar preservation and honest precision diagnostics."""

import unittest
from statistics import NormalDist

import numpy as np
import pandas as pd

from commodity_factor_lab.factor_evidence import SIGNALS, common_sample, precision_summary


def fixture():
    rng = np.random.default_rng(76)
    dates = pd.bdate_range("2023-01-02", periods=8)
    curves, labels = [], []
    for day in dates:
        for i in range(10):
            curves.append(
                {
                    "date": day,
                    "code": f"p{i}",
                    "momentum_126d": rng.normal(),
                    "carry": rng.normal(),
                    "basis_momentum": rng.normal(),
                    "inventory_scarcity": rng.normal(),
                }
            )
            labels.append({"date": day, "code": f"p{i}", "target": rng.normal(), "status": "ok"})
    settings = {
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
        "codes": [f"p{i}" for i in range(10)],
        "minimum_assets": 8,
    }
    return pd.DataFrame(curves), pd.DataFrame(labels), pd.DataFrame({"date": dates}), settings


class FactorEvidenceTests(unittest.TestCase):
    def test_missing_one_candidate_excludes_product_for_all_candidates(self):
        curves, labels, calendar, settings = fixture()
        curves.loc[curves.code.eq("p0"), "inventory_scarcity"] = np.nan
        panel, daily = common_sample(curves, labels, calendar, settings)
        self.assertTrue(daily.assets.eq(9).all())
        self.assertFalse(panel.loc[panel.code.eq("p0"), "used_in_common_estimate"].any())
        self.assertEqual(set(daily.signal), set(SIGNALS))

    def test_missing_future_label_is_a_measurement_exclusion(self):
        curves, labels, calendar, settings = fixture()
        labels.loc[labels.code.eq("p0"), ["target", "status"]] = [
            np.nan,
            "missing_same_contract_mark",
        ]
        panel, daily = common_sample(curves, labels, calendar, settings)
        self.assertTrue(panel.input_eligible.all())
        self.assertFalse(panel.loc[panel.code.eq("p0"), "measurement_eligible"].any())
        self.assertTrue(daily.assets.eq(9).all())

    def test_short_cross_section_keeps_whole_calendar_and_no_fake_zero(self):
        curves, labels, calendar, settings = fixture()
        curves.loc[
            curves.date.eq(calendar.date.iloc[3]) & curves.code.isin(["p0", "p1", "p2"]), "carry"
        ] = np.nan
        _, daily = common_sample(curves, labels, calendar, settings)
        self.assertEqual(len(daily), 3 * len(calendar))
        self.assertTrue(
            daily.loc[daily.date.eq(calendar.date.iloc[3]), "partial_rank_ic"].isna().all()
        )

    def test_degenerate_candidate_drops_date_for_all_signals(self):
        curves, labels, calendar, settings = fixture()
        curves.loc[curves.date.eq(calendar.date.iloc[2]), "inventory_scarcity"] = 1.0
        _, daily = common_sample(curves, labels, calendar, settings)
        self.assertTrue(
            daily.loc[daily.date.eq(calendar.date.iloc[2]), "partial_rank_ic"].isna().all()
        )

    def test_future_labels_do_not_change_earlier_daily_estimates(self):
        curves, labels, calendar, settings = fixture()
        _, before = common_sample(curves, labels, calendar, settings)
        labels.loc[labels.date.eq(calendar.date.iloc[-1]), "target"] *= -100
        _, after = common_sample(curves, labels, calendar, settings)
        pd.testing.assert_frame_equal(
            before.loc[before.date.lt(calendar.date.iloc[-1])],
            after.loc[after.date.lt(calendar.date.iloc[-1])],
        )

    def test_zero_lag_intervals_match_hand_calculation(self):
        result = precision_summary(pd.Series([1.0, 2.0, 3.0]), lags=0, min_obs=2)
        self.assertAlmostEqual(result["se"], np.sqrt(1 / 3))
        self.assertAlmostEqual(
            result["point_high"], 2 + NormalDist().inv_cdf(0.975) * np.sqrt(1 / 3)
        )
        self.assertGreater(result["simultaneous_high"], result["point_high"])

    def test_family_interval_widens_and_detectable_effect_increases(self):
        values = pd.Series(np.random.default_rng(12).normal(0, 0.1, 300))
        single = precision_summary(values, 0, 60, family=1)
        triple = precision_summary(values, 0, 60, family=3)
        self.assertGreater(triple["normal_planning_mde"], single["normal_planning_mde"])
        self.assertGreater(triple["simultaneous_high"], single["simultaneous_high"])

    def test_planning_diagnostic_uses_noise_not_observed_t_statistic(self):
        values = pd.Series(np.random.default_rng(4).normal(0, 0.1, 300))
        first = precision_summary(values, 0, 60)
        shifted = precision_summary(values + 0.2, 0, 60)
        self.assertAlmostEqual(first["normal_planning_mde"], shifted["normal_planning_mde"])
        self.assertNotAlmostEqual(first["t"], shifted["t"])
        self.assertAlmostEqual(first["power_at_resolution"], shifted["power_at_resolution"])

    def test_no_detectable_effect_within_correlation_range_is_missing(self):
        result = precision_summary(pd.Series([-100.0, 100.0, -100.0, 100.0]), lags=0, min_obs=2)
        self.assertTrue(np.isnan(result["normal_planning_mde"]))

    def test_degenerate_or_short_history_does_not_receive_intervals(self):
        for values in [pd.Series(np.ones(300)), pd.Series(np.arange(20))]:
            result = precision_summary(values)
            self.assertTrue(np.isnan(result["point_low"]))
            self.assertFalse(result["small_effect_bound_supported"])

    def test_invalid_settings_are_rejected(self):
        for kwargs in ({"family": 0}, {"alpha": 1}, {"resolution": 0}, {"power": 0.1}):
            with self.assertRaises(ValueError):
                precision_summary(pd.Series(np.arange(300)), **kwargs)

    def test_small_effect_bound_requires_a_narrow_whole_interval(self):
        noise = pd.Series(np.random.default_rng(192).normal(0, 1, 500))
        narrow = precision_summary(noise * 0.005, lags=0, min_obs=60)
        wide = precision_summary(noise, lags=0, min_obs=60)
        self.assertTrue(narrow["small_effect_bound_supported"])
        self.assertFalse(wide["small_effect_bound_supported"])


if __name__ == "__main__":
    unittest.main()
