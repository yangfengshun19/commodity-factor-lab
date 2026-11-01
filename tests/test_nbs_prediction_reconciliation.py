import importlib.util
import tempfile
import unittest
from pathlib import Path

from test_nbs_prediction import regression_fixture, source_fixture

from commodity_factor_lab.nbs_prediction import build_event_panel, recursive_forecasts

try:
    import statsmodels.api  # noqa: F401
except ImportError:
    HAS_REFERENCE = False
else:
    HAS_REFERENCE = True


@unittest.skipUnless(HAS_REFERENCE, "Install the dev extra for the independent OLS implementation")
class NbsPredictionReconciliationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "scripts/reconcile_nbs_prediction.py"
        spec = importlib.util.spec_from_file_location("nbs_prediction_audit", source)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_source_checker_detects_wrong_signal(self):
        frames = source_fixture()
        panel = build_event_panel(*frames)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for name, frame in zip(
                ("references", "curves", "labels", "bars", "calendar"), frames, strict=True
            ):
                path = root / f"{name}.csv"
                frame.to_csv(path, index=False)
                paths[name] = path
            original = self.module.Checkbook()
            self.module.audit_events(paths, panel, original)
            self.assertEqual(original.differences, [])
            changed = panel.copy()
            changed.loc[2, "reference_log_change"] += 0.1
            bad = self.module.Checkbook()
            self.module.audit_events(paths, changed, bad)
            self.assertTrue(
                any("reference log change" in item["check"] for item in bad.differences)
            )

    def test_other_ols_solver_agrees_and_detects_wrong_prediction(self):
        panel = regression_fixture()
        forecasts, parameters = recursive_forecasts(panel, "2020-01-01", "2021-12-31")
        good = self.module.Checkbook()
        self.module.audit_forecasts(panel, forecasts, parameters, good, "2020-01-01", "2021-12-31")
        self.assertEqual(good.differences, [])
        changed = forecasts.copy()
        first = changed.index[changed.prediction_status.eq("ok")][0]
        changed.loc[first, "prediction_augmented"] += 0.01
        bad = self.module.Checkbook()
        self.module.audit_forecasts(panel, changed, parameters, bad, "2020-01-01", "2021-12-31")
        self.assertTrue(any("prediction augmented" in item["check"] for item in bad.differences))

    def test_checker_detects_wrong_training_cutoff_and_membership(self):
        panel = regression_fixture()
        forecasts, parameters = recursive_forecasts(panel, "2020-01-01", "2021-12-31")
        first = forecasts.index[forecasts.prediction_status.eq("ok")][0]
        forecasts.loc[first, "training_max_exit"] = forecasts.loc[first, "date"]
        forecasts.loc[first, "training_periods_sha256"] = "wrong"
        bad = self.module.Checkbook()
        self.module.audit_forecasts(panel, forecasts, parameters, bad, "2020-01-01", "2021-12-31")
        self.assertTrue(any("latest target exit" in item["check"] for item in bad.differences))
        self.assertTrue(any("training membership" in item["check"] for item in bad.differences))
