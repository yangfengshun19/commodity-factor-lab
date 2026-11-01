import unittest

import numpy as np
import pandas as pd

from commodity_factor_lab.inference import hac_alpha, hac_mean

try:
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_hac
except ImportError:
    sm = None


@unittest.skipIf(sm is None, "Install the dev extra for the independent reference implementation")
class HACReferenceTests(unittest.TestCase):
    def test_mean_hac_matches_statsmodels_on_a_complete_calendar(self):
        rng = np.random.default_rng(27)
        values = pd.Series(rng.normal(0.001, 0.01, 500))
        values.iloc[1:] += 0.3 * values.iloc[:-1].to_numpy()
        fitted = sm.OLS(values, np.ones((len(values), 1))).fit()
        reference = cov_hac(fitted, nlags=10, use_correction=True)
        result = hac_mean(values, lags=10, min_obs=60)
        self.assertAlmostEqual(result["se"], np.sqrt(reference[0, 0]), places=13)

    def test_multi_benchmark_hac_matches_statsmodels(self):
        rng = np.random.default_rng(43)
        factors = pd.DataFrame(rng.normal(0, 0.01, (500, 2)), columns=["market", "momentum"])
        response = 0.001 + 0.4 * factors.market - 0.2 * factors.momentum + rng.normal(0, 0.005, 500)
        fitted = sm.OLS(response, sm.add_constant(factors)).fit()
        reference = cov_hac(fitted, nlags=10, use_correction=True)
        result = hac_alpha(response, factors, lags=10)
        self.assertAlmostEqual(
            result["t"], fitted.params.iloc[0] / np.sqrt(reference[0, 0]), places=10
        )
