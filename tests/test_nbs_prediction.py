import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from commodity_factor_lab.contract_research import future_contract_labels
from commodity_factor_lab.nbs_prediction import (
    CONTROLS,
    FEATURES,
    build_event_panel,
    forecast_metrics,
    period_number,
    prediction_plan,
    recursive_forecasts,
    register_experiment,
)

ROOT = Path(__file__).resolve().parents[1]


def source_fixture():
    dates = pd.bdate_range("2020-01-01", periods=100)
    calendar = pd.DataFrame(
        {
            "date": dates,
            "signal_cutoff": [f"{d:%Y-%m-%d}T12:30:00+00:00" for d in dates],
        }
    )
    i = np.arange(len(dates))
    curves = pd.DataFrame(
        {
            "date": dates,
            "code": "cu",
            "front_contract": np.where(i < 15, "cu2008", "cu2009"),
            "front_expiry": pd.Timestamp("2020-12-01"),
            "carry": np.sin(i),
            "momentum_126d": i / 100,
            "basis_momentum": np.cos(i),
            "reference_settle": np.where(i < 15, 100 + i, 1000 + 2 * i),
        }
    )
    bars = pd.DataFrame(
        [
            {"date": day, "code": "cu", "contract": contract, "settle": base + multiplier * j}
            for j, day in enumerate(dates)
            for contract, base, multiplier in (("cu2008", 100, 1), ("cu2009", 1000, 2))
        ]
    )
    labels = future_contract_labels(curves, bars, calendar)
    periods = ["2020-01-early", "2020-01-middle", "2020-01-late", "2020-02-early", "2020-02-middle"]
    references = pd.DataFrame(
        [
            {
                "period": period,
                "related_code": "cu",
                "reference_mapping_usable": True,
                "reference_price": 100 + j * 3,
                "comparability_key": "same",
                "source_declared_published_at": f"{dates[index]:%Y-%m-%d}T01:30:00+00:00",
                "first_signal_date": dates[index],
                "first_signal_cutoff": calendar.iloc[index].signal_cutoff,
                "source_url": "https://www.stats.gov.cn/synthetic",
                "source_sha256": "synthetic",
            }
            for j, (period, index) in enumerate(zip(periods, (3, 10, 17, 24, 90), strict=True))
        ]
    )
    return references, curves, labels, bars, calendar


def regression_fixture():
    rng = np.random.default_rng(34)
    dates = pd.bdate_range("2020-01-01", periods=130, freq="2B")
    features = rng.normal(size=(len(dates), len(FEATURES)))
    target = 0.002 + features @ np.array([0.01, -0.02, 0.03, 0.004, 0.017])
    frame = pd.DataFrame(features, columns=FEATURES)
    frame["target"] = target
    frame["date"] = dates
    frame["exit_date"] = dates + pd.offsets.BDay(21)
    frame["entry_date"] = dates + pd.offsets.BDay(1)
    frame["period"] = [f"event-{j:03}" for j in range(len(dates))]
    frame["period_number"] = np.arange(len(dates))
    frame["code"] = "cu"
    frame["input_status"] = "ok"
    frame["label_status"] = "ok"
    return frame


class NbsPredictionTests(unittest.TestCase):
    def test_adjacent_period_number_across_year(self):
        self.assertEqual(period_number("2021-01-early") - period_number("2020-12-late"), 1)
        with self.assertRaises(ValueError):
            period_number("2020-13-early")

    def test_reference_change_and_same_contract_past_control(self):
        refs, curves, labels, bars, calendar = source_fixture()
        panel = build_event_panel(refs, curves, labels, bars, calendar)
        row = panel.iloc[2]
        self.assertAlmostEqual(row.reference_log_change, np.log(106 / 103))
        # The current selected contract is cu2009; cu2008's old level must not enter.
        self.assertAlmostEqual(row.futures_since_previous_release, 1034 / 1020 - 1)
        self.assertEqual(panel.iloc[0].input_status, "no_adjacent_published_reference")
        self.assertTrue(panel.target_verified.all())

    def test_cancelled_period_and_specification_break_are_not_bridged(self):
        refs, *others = source_fixture()
        refs = refs.drop(index=1).copy()
        refs.loc[3, "comparability_key"] = "new"
        panel = build_event_panel(refs, *others)
        self.assertEqual(panel.iloc[1].input_status, "no_adjacent_published_reference")
        self.assertEqual(panel.iloc[2].input_status, "comparability_break")
        self.assertTrue(panel.iloc[1:3].reference_log_change.isna().all())

    def test_publication_not_backdated_and_target_tamper_rejected(self):
        refs, curves, labels, bars, calendar = source_fixture()
        refs.loc[1, "source_declared_published_at"] = "2020-01-15T13:00:00+00:00"
        with self.assertRaisesRegex(ValueError, "first cutoff"):
            build_event_panel(refs, curves, labels, bars, calendar)
        refs, curves, labels, bars, calendar = source_fixture()
        labels.loc[labels.date.eq(refs.iloc[1].first_signal_date), "target"] += 0.1
        with self.assertRaisesRegex(ValueError, "source marks"):
            build_event_panel(refs, curves, labels, bars, calendar)

    def test_invalid_clock_cutoff_and_changed_label_contract_rejected(self):
        refs, curves, labels, bars, calendar = source_fixture()
        refs.loc[1, "first_signal_cutoff"] = "2020-01-15T11:30:00+00:00"
        with self.assertRaisesRegex(ValueError, "cutoff differs"):
            build_event_panel(refs, curves, labels, bars, calendar)
        refs, curves, labels, bars, calendar = source_fixture()
        labels.loc[labels.date.eq(refs.iloc[1].first_signal_date), "contract"] = "cu2009"
        with self.assertRaisesRegex(ValueError, "selected contract"):
            build_event_panel(refs, curves, labels, bars, calendar)

    def test_tail_input_remains_admissible_without_future_target(self):
        panel = build_event_panel(*source_fixture())
        row = panel.iloc[-1]
        self.assertEqual(row.input_status, "ok")
        self.assertEqual(row.label_status, "outside_future_calendar")
        self.assertTrue(np.isnan(row.target))

    def test_recursive_purge_scaling_and_parameter_reference(self):
        panel = regression_fixture()
        forecasts, parameters = recursive_forecasts(panel, "2020-01-01", "2021-12-31")
        first = forecasts.loc[forecasts.prediction_status.eq("ok")].iloc[0]
        train = panel.loc[panel.exit_date.lt(first.date)]
        self.assertEqual(first.training_events, 60)
        self.assertEqual(first.training_events, len(train))
        self.assertLess(first.training_max_exit, first.date)
        self.assertAlmostEqual(first.prediction_augmented, first.target, places=13)
        actual = parameters.loc[
            parameters.period.eq(first.period) & parameters.model.eq("augmented")
        ]
        np.testing.assert_allclose(
            actual.iloc[1:].training_center, train[list(FEATURES)].mean().to_numpy(), atol=1e-14
        )
        design = np.column_stack([np.ones(len(train)), train[list(CONTROLS)]])
        reference = (
            np.r_[1.0, first[list(CONTROLS)].to_numpy(dtype=float)]
            @ np.linalg.lstsq(design, train.target, rcond=None)[0]
        )
        self.assertAlmostEqual(first.prediction_futures, reference, places=13)

    def test_current_and_future_labels_cannot_change_past_forecasts(self):
        panel = regression_fixture()
        original, _ = recursive_forecasts(panel, "2020-01-01", "2021-12-31")
        row = original.loc[original.prediction_status.eq("ok")].iloc[0]
        changed = panel.copy()
        not_yet_known = changed.exit_date.ge(row.date)
        changed.loc[not_yet_known, "target"] = 999
        changed.loc[changed.date.gt(row.date), list(FEATURES)] *= 100000
        replay, _ = recursive_forecasts(changed, "2020-01-01", "2021-12-31")
        replay_row = replay.loc[replay.period.eq(row.period)].iloc[0]
        self.assertEqual(row.prediction_augmented, replay_row.prediction_augmented)
        self.assertEqual(row.prediction_futures, replay_row.prediction_futures)
        self.assertEqual(row.training_periods_sha256, replay_row.training_periods_sha256)
        changed.loc[changed.date.eq(row.date), "target"] = np.nan
        replay, _ = recursive_forecasts(changed, "2020-01-01", "2021-12-31")
        unscored = replay.loc[replay.period.eq(row.period)].iloc[0]
        self.assertEqual(unscored.prediction_status, "ok")
        self.assertEqual(unscored.score_status, "forecast_without_observed_target")
        self.assertEqual(row.prediction_augmented, unscored.prediction_augmented)

    def test_constant_feature_is_not_force_fitted(self):
        panel = regression_fixture()
        panel["carry"] = 0.1
        forecasts, _ = recursive_forecasts(panel, "2020-01-01", "2021-12-31")
        self.assertFalse(forecasts.prediction_status.eq("ok").any())
        self.assertTrue(forecasts.prediction_status.eq("constant_training_feature").any())

    def test_release_weighting_and_all_baselines_use_same_rows(self):
        frame = pd.DataFrame(
            [
                {
                    "period": "a",
                    "date": pd.Timestamp("2023-01-01"),
                    "code": "cu",
                    "score_status": "scored",
                    "loss_futures": 10.0,
                    "loss_augmented": 4.0,
                    "loss_training_mean": 9.0,
                    "loss_zero": 9.0,
                },
                {
                    "period": "a",
                    "date": pd.Timestamp("2023-01-01"),
                    "code": "al",
                    "score_status": "scored",
                    "loss_futures": 10.0,
                    "loss_augmented": 4.0,
                    "loss_training_mean": 9.0,
                    "loss_zero": 9.0,
                },
                {
                    "period": "b",
                    "date": pd.Timestamp("2023-02-01"),
                    "code": "cu",
                    "score_status": "scored",
                    "loss_futures": 2.0,
                    "loss_augmented": 8.0,
                    "loss_training_mean": 9.0,
                    "loss_zero": 9.0,
                },
            ]
        )
        metrics, events = forecast_metrics(frame)
        self.assertEqual(len(events), 2)
        self.assertAlmostEqual(metrics.iloc[0].mse_augmented, 6.0)
        self.assertAlmostEqual(metrics.iloc[0].r2_vs_futures, 0.0)
        self.assertEqual(metrics.iloc[0].scored_rows, 3)

    def test_new_descriptive_experiment_is_reserved_in_multiplicity(self):
        prior = pd.DataFrame({"analysis": [str(i) for i in range(14)], "hac_p": [0.001] * 14})
        registry = register_experiment(prior, 90)
        self.assertEqual(len(registry), 15)
        self.assertTrue(np.isnan(registry.iloc[-1].hac_p))
        self.assertTrue(np.isnan(registry.iloc[-1].family_15_by_q))
        expected = 0.001 * 15 * sum(1 / j for j in range(1, 16)) / 14
        self.assertAlmostEqual(registry.iloc[0].family_15_by_q, expected)
        self.assertEqual(registry.iloc[0].hac_p, prior.iloc[0].hac_p)

    def test_protocol_rejects_lookahead_relaxation_and_source_changes(self):
        template = (ROOT / "configs/nbs_prediction.toml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            # Point every input into its own data folder and register synthetic content.
            import tomllib

            config = root / "configs/prediction.toml"
            raw = tomllib.loads(template)
            for key, old in raw["paths"].items():
                if key == "output":
                    continue
                path = root / "data" / key / "input.csv"
                path.parent.mkdir(parents=True)
                path.write_text(key, encoding="utf-8")
                template = template.replace(f'{key} = "{old}"', f'{key} = "data/{key}/input.csv"')
                template = template.replace(
                    raw["input_hashes"][key], hashlib.sha256(path.read_bytes()).hexdigest()
                )
            config.write_text(template, encoding="utf-8")
            prediction_plan(config)
            config.write_text(
                template.replace(
                    "training_target_exit_strictly_before_signal_date = true",
                    "training_target_exit_strictly_before_signal_date = false",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "fixed descriptive"):
                prediction_plan(config)
            config.write_text(template, encoding="utf-8")
            (root / "data/bars/input.csv").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input changed"):
                prediction_plan(config)
