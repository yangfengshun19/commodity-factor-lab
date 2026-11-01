import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from commodity_factor_lab.factor_audit import arithmetic_check, load_decisions
from commodity_factor_lab.factor_extension import evaluate_extension, verify_frozen_inputs
from commodity_factor_lab.inference import by_adjust
from commodity_factor_lab.positions import parse_position_ranks


class FactorRegistryTests(unittest.TestCase):
    def test_complete_registry_requires_every_indicator_once(self):
        registry = Path(__file__).resolve().parents[1] / "configs/factor_decisions.toml"
        rows = load_decisions(registry)
        self.assertEqual(len(rows), 26)
        with tempfile.TemporaryDirectory() as temporary:
            invalid = Path(temporary) / "invalid.toml"
            invalid.write_bytes(
                registry.read_bytes() + b'\n[[factors]]\nfamily="value"\nlegacy_id="Val_halfyear"\n'
            )
            with self.assertRaisesRegex(ValueError, "exactly once"):
                load_decisions(invalid)

    def test_retained_value_is_not_allowed_to_hide_missing_reference(self):
        result = arithmetic_check(
            pd.DataFrame({"legacy": [0.1, 0.2, np.nan]}),
            "legacy",
            pd.Series([0.1, np.nan, 0.3]),
            "fixture",
        )
        self.assertEqual(result["comparable_rows"], 1)
        self.assertEqual(result["legacy_present_reference_missing"], 1)
        self.assertEqual(result["reference_present_legacy_missing"], 1)


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(23)
        self.data = pd.DataFrame(
            {
                "date": np.repeat(pd.bdate_range("2023-01-02", periods=300), 10),
                **{key: rng.normal(size=3000) for key in ("new", "target", "c1", "c2", "c3")},
            }
        )
        self.study = dict(
            min_assets=8,
            hac_lags=40,
            holding_sessions=20,
            min_inference_obs=252,
            evaluation_start="2023-01-01",
        )
        self.hypotheses = [
            dict(
                name=name,
                controls=["c1", "c2", "c3"],
                direction=1,
                status="ready" if name == "new" else "gated_source_validation",
                reason="fixture",
            )
            for name in ("new", "missing1", "missing2")
        ]

    def test_gated_hypotheses_still_expand_the_correction_family(self):
        daily, summary = evaluate_extension(self.data, self.hypotheses, self.study)
        active = summary.iloc[0]
        self.assertEqual(active.days, 300)
        self.assertEqual(active.min_assets_used, 10)
        self.assertAlmostEqual(active.by_q, by_adjust([active.hac_p, np.nan, np.nan])[0])
        self.assertTrue(summary.iloc[1:].by_q.isna().all())
        self.assertTrue(summary.iloc[1:].days.eq(0).all())
        self.assertTrue(daily.loc[daily.signal.ne("new"), "partial_rank_ic"].isna().all())
        self.assertFalse(summary.claim_eligible.any())

    def test_seven_products_cannot_pass_the_fixed_cross_section(self):
        data = self.data.loc[self.data.groupby("date").cumcount().lt(7)]
        _, summary = evaluate_extension(data, self.hypotheses, self.study)
        self.assertTrue(summary.days.eq(0).all())

    def test_no_sign_reversal_or_missing_control(self):
        self.hypotheses[0]["direction"] = -1
        with self.assertRaisesRegex(ValueError, "sign reversal"):
            evaluate_extension(self.data, self.hypotheses, self.study)
        self.hypotheses[0]["direction"] = 1
        with self.assertRaisesRegex(ValueError, "source feature/control"):
            evaluate_extension(self.data.drop(columns="c3"), self.hypotheses, self.study)

    def test_frozen_input_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("curves.csv", "labels.csv", "run.json", "prior.csv"):
                (root / name).write_text("frozen", encoding="utf-8")
            digest = hashlib.sha256(b"frozen").hexdigest()
            config = {
                "paths": {"baseline_run": ".", "prior_exploration": "prior.csv"},
                "input_hashes": {
                    k: digest for k in ("curves", "labels", "run", "prior_exploration")
                },
            }
            self.assertEqual(len(verify_frozen_inputs(config, root)[1]), 4)
            (root / "labels.csv").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "labels"):
                verify_frozen_inputs(config, root)


class PositionRankTests(unittest.TestCase):
    def payload(self, rows=None, **changes):
        if rows is None:
            rows = [
                {
                    "INSTRUMENTID": "cu2404 ",
                    "RANK": i,
                    "PARTICIPANTID2": f"{i:04}",
                    "CJ2": 100 + i,
                    "PARTICIPANTID3": f"{i + 30:04}",
                    "CJ3": 50 + i,
                }
                for i in range(1, 21)
            ]
        return json.dumps(
            {
                "report_date": "20240130",
                "update_date": "20240130 19:57:20",
                "o_cursor": rows,
                **changes,
            }
        ).encode()

    def test_only_actual_contract_top20_counts_and_clock_is_utc(self):
        rows = json.loads(self.payload())["o_cursor"]
        rows += [{"INSTRUMENTID": "cuall", "RANK": -1}, {"INSTRUMENTID": "cu2404C90000", "RANK": 1}]
        result = parse_position_ranks(self.payload(rows), "20240130")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].status, "complete_top20")
        self.assertEqual(result.iloc[0].long_top20, 2210)
        self.assertEqual(str(result.iloc[0].available_at), "2024-01-30 11:57:20+00:00")

    def test_missing_duplicate_or_blank_rank_is_not_a_zero_position(self):
        rows = json.loads(self.payload())["o_cursor"]
        variants = [rows[:-1], rows + [rows[-1]], [dict(r) for r in rows], [dict(r) for r in rows]]
        variants[2][0]["CJ2"] = ""
        variants[3][0]["PARTICIPANTID2"] = variants[3][1]["PARTICIPANTID2"]
        for variant in variants:
            with self.subTest(variant=variant[0]):
                result = parse_position_ranks(self.payload(variant), "20240130")
                self.assertEqual(result.iloc[0].status, "incomplete_or_invalid_ranks")
                self.assertTrue(pd.isna(result.iloc[0].long_top20))
                self.assertTrue(pd.isna(result.iloc[0].short_top20))

    def test_wrong_report_or_date_only_clock_is_rejected(self):
        for changes in ({"report_date": "20240131"}, {"update_date": "20240130"}):
            with self.assertRaises(ValueError):
                parse_position_ranks(self.payload(**changes), "20240130")
