import unittest

import numpy as np
import pandas as pd

from commodity_factor_lab.inference import by_adjust, hac_alpha, hac_mean, paired_block_interval


class InferenceTests(unittest.TestCase):
    def test_intercept_only_hac_zero_lag_matches_sample_mean_se(self):
        result = hac_mean(pd.Series([1.0, 2.0, 3.0]), lags=0, min_obs=2)
        self.assertAlmostEqual(result["mean"], 2)
        self.assertAlmostEqual(result["se"], np.sqrt(1 / 3))

    def test_missing_calendar_slots_are_not_compressed(self):
        missing = hac_mean(pd.Series([1.0, np.nan, 2.0, np.nan, 4.0]), 1, 2)
        compressed = hac_mean(pd.Series([1.0, 2.0, 4.0]), 1, 2)
        self.assertAlmostEqual(missing["mean"], compressed["mean"])
        self.assertNotAlmostEqual(missing["se"], compressed["se"])

    def test_short_samples_do_not_receive_a_significance_pvalue(self):
        result = hac_mean(pd.Series(np.arange(35)), min_obs=60)
        self.assertTrue(np.isnan(result["p"]))
        self.assertEqual(result["status"], "insufficient_history")

    def test_constant_profit_is_not_reported_as_infinite_certainty(self):
        self.assertTrue(np.isnan(hac_mean(pd.Series(np.ones(100)), min_obs=60)["p"]))

    def test_by_qvalues_use_the_declared_family(self):
        np.testing.assert_allclose(
            by_adjust([0.01, 0.04, 0.03]), [0.055, 0.073333333333, 0.073333333333]
        )
        expanded = by_adjust([0.01, 0.04, 0.03, np.nan])
        self.assertGreater(expanded[0], 0.055)
        self.assertTrue(np.isnan(expanded[-1]))

    def test_invalid_pvalues_are_rejected(self):
        with self.assertRaises(ValueError):
            by_adjust([1.1])

    def test_hac_benchmark_regression_recovers_an_intercept(self):
        x = pd.Series(np.sin(np.arange(500)) * 0.01)
        residual = pd.Series(np.cos(np.arange(500) * 1.7) * 0.003)
        result = hac_alpha(0.001 + 0.4 * x + residual, pd.DataFrame({"market": x}))
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["annual_intercept"], 0.252, delta=0.004)

    def test_rank_deficient_benchmark_model_is_not_silently_fitted(self):
        values = pd.Series(np.sin(np.arange(300)))
        result = hac_alpha(values, pd.DataFrame({"a": values, "b": values}))
        self.assertEqual(result["status"], "rank_deficient")

    def test_paired_bootstrap_uses_shared_draws_and_is_reproducible(self):
        values = pd.Series(np.sin(np.arange(300)) * 0.01)
        result = paired_block_interval(values, values, repetitions=100)
        self.assertEqual(result["low"], 0)
        self.assertEqual(result["high"], 0)
        self.assertEqual(result, paired_block_interval(values, values, repetitions=100))

    def test_bootstrap_refuses_to_hide_missing_paired_dates(self):
        values = pd.Series(np.ones(300))
        values.iloc[20] = np.nan
        self.assertEqual(
            paired_block_interval(values, values)["status"], "incomplete_paired_calendar"
        )
