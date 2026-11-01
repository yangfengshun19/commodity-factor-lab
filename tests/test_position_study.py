import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from commodity_factor_lab.position_ingest import ENDPOINT, fetch_positions
from commodity_factor_lab.position_study import qualify_positions, same_contract_position_panel


class SameContractPositionTests(unittest.TestCase):
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
                    "net_top20": base + i * slope,
                    "open_interest": 10000,
                    "status": "complete_top20",
                }
                for i, date in enumerate(self.dates)
                for contract, base, slope in (("cu2404", -3000, 100), ("cu2405", 5000, 20))
            ]
        )
        self.curves = pd.DataFrame(
            {"date": self.dates, "code": "cu", "front_contract": ["cu2404"] * 20 + ["cu2405"] * 5}
        )

    def test_roll_uses_the_current_contracts_unselected_past(self):
        result = same_contract_position_panel(self.qualified, self.curves, self.calendar)
        last = result.iloc[-1]
        self.assertAlmostEqual(last.normalized_position_flow, 0.04)
        self.assertEqual(last.net_today, 5480)
        self.assertEqual(last.net_lag20, 5080)
        self.assertEqual(last.contract, "cu2405")
        self.assertEqual(last.complete_window_sessions, 21)
        self.assertEqual(result.iloc[19].status, "warmup")

    def test_missing_middle_session_cannot_be_filled_or_calendar_compressed(self):
        keep = ~((self.qualified.contract == "cu2405") & (self.qualified.date == self.dates[5]))
        result = same_contract_position_panel(self.qualified.loc[keep], self.curves, self.calendar)
        self.assertTrue(pd.isna(result.iloc[-1].normalized_position_flow))
        self.assertEqual(result.iloc[-1].complete_window_sessions, 20)
        self.assertEqual(result.iloc[-1].lookback_date, self.dates[4])

    def test_unknown_selected_contract_is_missing(self):
        self.curves.loc[24, "front_contract"] = "cu2406"
        result = same_contract_position_panel(self.qualified, self.curves, self.calendar)
        self.assertEqual(result.iloc[-1].status, "not_disclosed")
        self.assertTrue(pd.isna(result.iloc[-1].normalized_position_flow))

    def test_publication_and_single_sided_oi_are_required(self):
        date = self.dates[0]
        ranks = pd.DataFrame(
            [
                {
                    "date": date,
                    "contract": contract,
                    "status": "complete_top20",
                    "long_top20": long,
                    "short_top20": short,
                    "available_at": date.tz_localize("UTC") + pd.Timedelta(clock, unit="h"),
                }
                for contract, long, short, clock in (
                    ("cu2404", 80, 60, 12),
                    ("cu2405", 80, 60, 13),
                    ("cu2406", 120, 110, 12),
                    ("cu2407", 80, 60, 12),
                )
            ]
        )
        bars = pd.DataFrame(
            {"date": [date] * 4, "contract": ranks.contract, "open_interest": [100, 100, 100, 0]}
        )
        result = qualify_positions(ranks, bars, self.calendar).set_index("contract")
        self.assertEqual(result.loc["cu2404", "net_top20"], 20)
        self.assertEqual(result.loc["cu2405", "status"], "published_after_cutoff")
        self.assertEqual(result.loc["cu2406", "status"], "side_total_exceeds_single_sided_oi")
        self.assertEqual(result.loc["cu2407", "status"], "missing_or_invalid_oi")
        self.assertTrue(result.loc[["cu2405", "cu2406", "cu2407"], "net_top20"].isna().all())


class RankingCollectionTests(unittest.TestCase):
    def plan(self, root):
        return (
            {
                "collection": {
                    "maximum_attempts_per_date": 2,
                    "maximum_response_bytes": 8000000,
                    "minimum_interval_seconds": 0.8,
                }
            },
            {"raw": root / "raw", "seed": root / "no_seed"},
            ["20240130", "20240131", "20240201"],
        )

    def response(self, date):
        return json.dumps(
            {"report_date": date, "update_date": date + " 19:00:00", "o_cursor": []}
        ).encode()

    def test_resume_reuses_verified_bytes_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.toml"
            config.write_text("frozen", encoding="utf-8")
            with (
                patch(
                    "commodity_factor_lab.position_ingest.position_plan",
                    return_value=self.plan(root),
                ),
                patch("commodity_factor_lab.position_ingest.time.sleep"),
                patch(
                    "commodity_factor_lab.position_ingest.PublicRankingConnection.read",
                    side_effect=lambda date, maximum: (200, self.response(date)),
                ) as read,
            ):
                first = fetch_positions(config, limit=1)
                self.assertEqual(first["pending"], 2)
                self.assertEqual(read.call_count, 1)
                second = fetch_positions(config)
                self.assertEqual(second["pending"], 0)
                self.assertEqual(read.call_count, 3)
                (root / "raw/responses/20240130.json").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "changed"):
                    fetch_positions(config)

    def test_access_limit_is_preserved_and_stops_further_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.toml"
            config.write_text("frozen", encoding="utf-8")
            with (
                patch(
                    "commodity_factor_lab.position_ingest.position_plan",
                    return_value=self.plan(root),
                ),
                patch("commodity_factor_lab.position_ingest.time.sleep"),
                patch(
                    "commodity_factor_lab.position_ingest.PublicRankingConnection.read",
                    return_value=(403, b"access denied"),
                ) as read,
            ):
                result = fetch_positions(config)
                self.assertEqual(result["completed"], 1)
                self.assertEqual(read.call_count, 1)
            receipt = json.loads((root / "raw/manifest.json").read_text())
            self.assertEqual(receipt["records"][0]["attempts"][0]["http_status"], 403)
            self.assertTrue((root / "raw/responses/20240130.attempt1.bin").exists())
            self.assertEqual(receipt["records"][0]["url"], ENDPOINT.format(date="20240130"))
