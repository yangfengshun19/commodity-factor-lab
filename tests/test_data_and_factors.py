import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from commodity_factor_lab.catalog import code_from_filename
from commodity_factor_lab.config import ResearchConfig, load_config
from commodity_factor_lab.data import load_family, normalize_snapshot
from commodity_factor_lab.factors import price_factors


def bars(length=280):
    prices = np.arange(length, dtype=float) + 100
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2023-01-02", periods=length),
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
        }
    )


class DataTests(unittest.TestCase):
    def test_unnamed_date_header_is_repaired_but_numeric_index_is_not(self):
        frame = bars(3)
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        frame = frame.rename(columns={"date": "Unnamed: 0"})
        cleaned, info = normalize_snapshot(frame, ("factor",))
        self.assertTrue(info["date_header_repaired"])
        self.assertEqual(len(cleaned), 3)
        frame["Unnamed: 0"] = [0, 1, 2]
        with self.assertRaises(ValueError):
            normalize_snapshot(frame, ())

    def test_epoch_rows_are_excluded_without_guessing_a_date(self):
        frame = bars(3)
        frame.loc[0, "date"] = pd.Timestamp("1970-01-01")
        cleaned, info = normalize_snapshot(frame, ())
        self.assertEqual(info["invalid_date_rows"], 1)
        self.assertEqual(len(cleaned), 2)

    def test_invalid_prices_keep_the_calendar_and_invalidate_signals(self):
        frame = bars(3)
        frame["factor"] = 1
        frame.loc[1, "open"] = 0
        cleaned, info = normalize_snapshot(frame, ("factor",))
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(info["invalid_ohlc_rows"], 1)
        self.assertTrue(pd.isna(cleaned.iloc[1]["open"]))
        self.assertTrue(pd.isna(cleaned.iloc[1]["factor"]))
        self.assertEqual(frame.loc[1, "open"], 0)

    def test_duplicate_dates_are_not_silently_deduplicated(self):
        frame = bars(3)
        frame.loc[1, "date"] = frame.loc[0, "date"]
        with self.assertRaises(ValueError):
            normalize_snapshot(frame, ())

    def test_bad_later_close_does_not_cancel_a_valid_opening_mark(self):
        frame = bars(3)
        frame["factor"] = 1
        frame.loc[1, "close"] = -1
        cleaned, info = normalize_snapshot(frame, ("factor",))
        self.assertEqual(info["invalid_ohlc_rows"], 1)
        self.assertEqual(cleaned.iloc[1]["open"], 101)
        self.assertTrue(pd.isna(cleaned.iloc[1]["factor"]))

    def test_file_codes_are_normalized_and_unknowns_rejected(self):
        self.assertEqual(code_from_filename("螺纹钢_RB_factors.csv"), "rb")
        with self.assertRaises(ValueError):
            code_from_filename("panel.csv")

    def test_financial_futures_are_excluded_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "trend").mkdir()
            bars(3).to_csv(root / "trend/Index_IF_factors.csv", index=False)
            cfg = ResearchConfig(data=root, output=root.parent / "outputs")
            loaded = load_family(cfg, "trend")
            self.assertTrue(loaded.panel.empty)
            self.assertEqual(loaded.quality[0]["status"], "excluded_financial")


class FactorTests(unittest.TestCase):
    def test_rolling_warmup_is_not_filled_with_partial_history(self):
        factors = price_factors(bars())
        self.assertTrue(factors["Val_oneyear"].iloc[:252].isna().all())
        self.assertAlmostEqual(factors["Val_oneyear"].iloc[252], 352 / 100 - 1)
        self.assertTrue(factors["vol_std"].iloc[:20].isna().all())

    def test_future_bars_do_not_change_past_factors(self):
        frame = bars()
        reference = price_factors(frame)
        for col in ("open", "high", "low", "close"):
            frame.loc[250:, col] *= 2
        changed = price_factors(frame)
        pd.testing.assert_frame_equal(reference.iloc[:250], changed.iloc[:250])

    def test_unsorted_bars_are_sorted_before_calculation(self):
        frame = bars()
        pd.testing.assert_frame_equal(price_factors(frame), price_factors(frame.iloc[::-1]))

    def test_nonpositive_prices_are_rejected(self):
        frame = bars()
        frame.loc[0, "close"] = 0
        with self.assertRaises(ValueError):
            price_factors(frame)


class ConfigTests(unittest.TestCase):
    def test_invalid_group_size_and_leaking_output_are_rejected(self):
        cfg = ResearchConfig(data=Path("data/input"), output=Path("reports/output"))
        with self.assertRaises(ValueError):
            replace(cfg, min_assets=3)
        with self.assertRaises(ValueError):
            replace(cfg, output=cfg.data / "generated")
        with self.assertRaises(ValueError):
            replace(cfg, fee_bps=float("nan"))

    def test_config_paths_are_independent_of_working_directory(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs/demo.toml")
        self.assertEqual(cfg.data, root / "data/demo")


if __name__ == "__main__":
    unittest.main()
