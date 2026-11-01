"""Independent library references and deliberate corruptions of saved evidence."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from test_factor_evidence import fixture

from commodity_factor_lab.factor_evidence import common_sample, precision_summary
from commodity_factor_lab.inference import by_adjust

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reconcile_factor_evidence.py"
SPEC = importlib.util.spec_from_file_location("evidence_auditor", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class EvidenceReferenceTests(unittest.TestCase):
    def test_calendar_gaps_match_independent_library_covariance(self):
        rng = np.random.default_rng(309)
        values = pd.Series(rng.normal(0, 0.1, 500))
        values.iloc[30:46] = np.nan
        actual = precision_summary(values, 10, 60)
        expected = AUDIT.reference_precision(values, 10)
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key], places=11)

    def test_planning_mde_matches_independent_root_solver(self):
        values = pd.Series(np.random.default_rng(211).normal(0, 0.5, 300))
        result = precision_summary(values, 20, 60)
        reference = AUDIT.reference_precision(values, 20)
        self.assertAlmostEqual(
            result["normal_planning_mde"], reference["normal_planning_mde"], places=10
        )

    def test_auditor_detects_changed_daily_ic_and_confidence_limit(self):
        curves, labels, calendar, settings = fixture()
        a = settings | {
            "alpha": 0.05,
            "current_family_size": 3,
            "resolution_target_rank_ic": 0.05,
            "planning_power": 0.8,
            "registered_after": 20,
        }
        panel, daily = common_sample(curves, labels, calendar, settings)
        rows = [
            {"signal": signal, "primary": True, **precision_summary(group.partial_rank_ic, 1, 2)}
            for signal, group in daily.groupby("signal")
        ]
        summary = pd.DataFrame(rows)
        registry = pd.DataFrame({"hac_p": [0.01] + [np.nan] * 19})
        registry["family_20_by_q"] = by_adjust(registry.hac_p)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, frame in {
                "curves": curves,
                "labels": labels,
                "calendar": calendar,
                "common_panel": panel,
                "common_daily": daily,
                "primary": summary,
                "lag_sensitivity": summary,
                "registered_exploration": registry,
            }.items():
                frame.to_csv(root / f"{name}.csv", index=False)
            run = {
                "protocol": {
                    "analysis": a,
                    "paths": {name: f"{name}.csv" for name in ("curves", "labels", "calendar")},
                }
            }
            (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
            self.assertEqual(AUDIT.verify(root, root)["differences"], [])
            changed = summary.copy()
            changed.loc[0, "simultaneous_low"] += 0.1
            changed.to_csv(root / "primary.csv", index=False)
            self.assertTrue(
                any(
                    "/primary/simultaneous_low" in e
                    for e in AUDIT.verify(root, root)["differences"]
                )
            )
            summary.to_csv(root / "primary.csv", index=False)
            daily.loc[0, "partial_rank_ic"] += 0.1
            daily.to_csv(root / "common_daily.csv", index=False)
            self.assertTrue(any("/IC" in e for e in AUDIT.verify(root, root)["differences"]))


if __name__ == "__main__":
    unittest.main()
