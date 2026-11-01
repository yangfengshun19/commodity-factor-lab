import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from commodity_factor_lab.contract_demo import fixed_budget_orders
from commodity_factor_lab.contract_research import (
    future_contract_labels,
    partial_rank_ic,
    verify_receipt,
)
from commodity_factor_lab.fundamentals import seasonal_inventory
from commodity_factor_lab.official import (
    fetch_metadata,
    parse_annual_excel,
    parse_contract_metadata,
)
from commodity_factor_lab.warrants import parse_warrants


class ProviderTests(unittest.TestCase):
    def test_downloaded_metadata_has_a_consumable_normalized_receipt(self):
        payload = {
            "report_date": "20240401",
            "update_date": "20240401 16:05:01",
            "ContractBaseInfo": [
                {"INSTRUMENTID": "cu2404", "EXPIREDATE": "20240415", "OPENDATE": "20230418"}
            ],
        }
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "commodity_factor_lab.official.read_public",
                return_value=json.dumps(payload).encode(),
            ),
            patch("commodity_factor_lab.official.time.sleep"),
        ):
            output = Path(temporary) / "valid"
            fetch_metadata(["20240401"], output)
            receipt = verify_receipt(output / "metadata.csv")
            self.assertEqual(receipt["receipt"]["kind"], "official_contract_metadata")
            self.assertEqual(receipt["receipt"]["records"][0]["status"], "ok")

    def test_wrong_historical_report_is_preserved_but_not_accepted_as_metadata(self):
        payload = {"report_date": "20240402"}
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "commodity_factor_lab.official.read_public",
                return_value=json.dumps(payload).encode(),
            ),
            patch("commodity_factor_lab.official.time.sleep"),
        ):
            output = Path(temporary) / "wrong"
            fetch_metadata(["20240401"], output)
            self.assertFalse((output / "metadata.csv").exists())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["records"][0]["status"], "failed")
            self.assertIn("sha256", manifest["records"][0])
            self.assertTrue((output / "responses/20240401.json").exists())

    def test_file_update_clock_is_used_for_new_metadata_schema(self):
        payload = {
            "update_date": "20240401 16:05:01",
            "ContractBaseInfo": [
                {"INSTRUMENTID": "cu2404", "EXPIREDATE": "20240415", "OPENDATE": "20230418"}
            ],
        }
        result = parse_contract_metadata(json.dumps(payload).encode(), "test")
        self.assertEqual(str(result.iloc[0].metadata_available_at), "2024-04-01 08:05:01+00:00")
        self.assertEqual(result.iloc[0].timestamp_source, "file_update")

    def test_record_update_cannot_predate_the_file_becoming_available(self):
        payload = {
            "update_date": "20240401 16:05:01",
            "ContractBaseInfo": [
                {
                    "INSTRUMENTID": "cu2404",
                    "EXPIREDATE": "20240415",
                    "OPENDATE": "20230418",
                    "UPDATE_DATE": "2024-04-01 15:30:00",
                }
            ],
        }
        result = parse_contract_metadata(json.dumps(payload).encode(), "test")
        self.assertEqual(str(result.iloc[0].metadata_available_at), "2024-04-01 08:05:01+00:00")

    def test_date_only_metadata_does_not_become_available_at_midnight(self):
        payload = {
            "update_date": "20240401",
            "ContractBaseInfo": [
                {"INSTRUMENTID": "cu2404", "EXPIREDATE": "20240415", "OPENDATE": "20230418"}
            ],
        }
        result = parse_contract_metadata(json.dumps(payload).encode(), "test")
        self.assertTrue(pd.isna(result.iloc[0].metadata_available_at))
        self.assertEqual(result.iloc[0].timing_status, "unknown_clock")

    def test_merged_contract_identity_is_filled_but_missing_price_is_not(self):
        raw = pd.DataFrame(
            [
                [
                    "合约",
                    "交易日期",
                    "开盘价",
                    "最高价",
                    "最低价",
                    "收盘价",
                    "结算价",
                    "前结算",
                    "成交量",
                    "持仓量",
                ],
                ["cu2402", "20240102", 100, 102, 99, 101, 101, 99, 1000, 2000],
                [None, "20240103", None, 103, 98, 102, 102, 101, 2000, 3000],
                ["cu2402C900", "20240102", 1, 2, 1, 2, 2, 1, 3, 4],
                [None, "20240103", 2, 3, 2, 3, 3, 2, 3, 4],
            ]
        )
        with (
            patch("commodity_factor_lab.official.pd.ExcelFile") as book,
            patch("commodity_factor_lab.official.pd.read_excel", return_value=raw),
        ):
            book.return_value.sheet_names = ["official"]
            result = parse_annual_excel(b"fixture", ("cu",))
        self.assertEqual(result.contract.tolist(), ["cu2402", "cu2402"])
        self.assertTrue(np.isnan(result.iloc[1].open))
        self.assertEqual(result.iloc[1].settle, 102)

    def test_warehouse_grand_total_does_not_double_count_warehouses_and_subtotals(self):
        obj = {
            "report_date": "20200103",
            "update_date": "20200103 20:00:47",
            "o_cursor": [
                {"VARNAME": "铜$$COPPER", "WHABBRNAME": label, "WRTWGHTS": value, "WGHTUNIT": "2"}
                for label, value in [
                    ("仓库A", "100"),
                    ("合计$$Subtotal", "100"),
                    ("完税商品总计", "100"),
                    ("总计$$Total", "100"),
                ]
            ],
        }
        result = parse_warrants(json.dumps(obj).encode(), "20200103", "test")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].value, 100)
        self.assertEqual(result.iloc[0].unit, "SHFE_WGHTUNIT_2")


class ForecastTests(unittest.TestCase):
    def test_renamed_warehouse_total_excludes_factory_warrants(self):
        payload = {
            "report_date": "20240102",
            "update_date": "2024-01-02 17:00:00",
            "o_cursor": [
                {"VARNAME": name, "WHABBRNAME": "总计", "WRTWGHTS": value, "WGHTUNIT": "2"}
                for name, value in [("螺纹钢仓库$$Rebar Warehouse", "100"), ("螺纹钢厂库", "300")]
            ],
        }
        frame = parse_warrants(json.dumps(payload).encode(), "20240102", "test")
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0].value, 100)
        self.assertEqual(frame.iloc[0].code, "rb")

    def test_expanded_warrant_universe_is_explicit_and_keeps_warehouse_measure(self):
        payload = {
            "report_date": "20240102",
            "update_date": "2024-01-02 17:00:00",
            "o_cursor": [
                {"VARNAME": n, "WHABBRNAME": "总计", "WRTWGHTS": v, "WGHTUNIT": "2"}
                for n, v in [
                    ("铜", 100),
                    ("石油沥青仓库", 200),
                    ("石油沥青厂库", 300),
                    ("纸浆仓库", 400),
                ]
            ],
        }
        raw = json.dumps(payload).encode()
        self.assertEqual(parse_warrants(raw, "20240102", "test").code.tolist(), ["cu"])
        expanded = parse_warrants(raw, "20240102", "test", ("cu", "bu", "sp", "au"))
        self.assertEqual(
            dict(zip(expanded.code, expanded.value, strict=True)),
            {"cu": 100.0, "bu": 200.0, "sp": 400.0},
        )
        self.assertNotIn("au", expanded.code.tolist())

    def test_prediction_target_keeps_signal_selected_contract(self):
        dates = pd.bdate_range("2023-01-02", periods=5)
        bars = pd.DataFrame(
            [
                {"date": d, "contract": c, "settle": p}
                for d in dates
                for c, p in [("near", 100), ("far", 200)]
            ]
        )
        curves = pd.DataFrame(
            {
                "date": dates,
                "code": "rb",
                "front_contract": ["near", "far", "far", "far", "far"],
                "front_expiry": pd.Timestamp("2023-06-01"),
            }
        )
        result = future_contract_labels(curves, bars, pd.DataFrame({"date": dates}), 1, 2)
        self.assertEqual(result.iloc[0].contract, "near")
        self.assertEqual(result.iloc[0].target, 0)
        self.assertEqual(result.iloc[-1].status, "outside_future_calendar")

    def test_labels_crossing_known_expiry_are_unavailable(self):
        dates = pd.bdate_range("2023-01-02", periods=5)
        bars = pd.DataFrame({"date": dates, "contract": "near", "settle": 100})
        curves = pd.DataFrame(
            {"date": dates, "code": "rb", "front_contract": "near", "front_expiry": dates[2]}
        )
        result = future_contract_labels(curves, bars, pd.DataFrame({"date": dates}), 1, 2)
        self.assertEqual(result.iloc[0].status, "expiry_before_target_end")
        self.assertTrue(np.isnan(result.iloc[0].target))

    def test_no_incremental_variation_is_not_reported_as_new_predictability(self):
        frame = pd.DataFrame(
            {"signal": range(6), "momentum": range(6), "target": [1, 3, 2, 5, 0, 4]}
        )
        result = partial_rank_ic(frame, "signal", ("momentum",))
        self.assertEqual(result["status"], "no_incremental_rank_variation")
        self.assertTrue(np.isnan(result["partial_rank_ic"]))

    def test_january_does_not_count_last_december_as_an_earlier_season(self):
        rows = [
            {
                "code": "rb",
                "period_date": f"{year}-01-02",
                "available_at": f"{year}-01-02T16:00:00+08:00",
                "value": value,
                "unit": "tonnes",
                "source": "test",
            }
            for year, value in [(2020, 10), (2021, 20), (2022, 30), (2023, 40)]
        ]
        calendar = pd.DataFrame(
            {"date": ["2023-01-02"], "signal_cutoff": ["2023-01-02T17:00:00+08:00"]}
        )
        before = seasonal_inventory(pd.DataFrame(rows), calendar, min_observations=3)
        rows.append(
            {
                "code": "rb",
                "period_date": "2022-12-30",
                "available_at": "2022-12-30T16:00:00+08:00",
                "value": 40,
                "unit": "tonnes",
                "source": "test",
            }
        )
        after = seasonal_inventory(pd.DataFrame(rows), calendar, min_observations=3)
        self.assertEqual(before.iloc[0].inventory_scarcity, after.iloc[0].inventory_scarcity)
        self.assertEqual(after.iloc[0].seasonal_observations, 3)

    def test_future_opening_price_cannot_change_order_size(self):
        dates = pd.bdate_range("2023-01-02", periods=5)
        codes = ["a", "b", "c", "d", "e", "f"]
        curves = pd.DataFrame(
            [
                {
                    "date": d,
                    "code": code,
                    "carry": i,
                    "momentum_126d": i,
                    "basis_momentum": i,
                    "front_contract": code,
                    "reference_settle": 100.0,
                    "signal_cutoff": f"{d:%Y-%m-%d}T17:00:00+08:00",
                }
                for d in dates
                for i, code in enumerate(codes)
            ]
        )
        bars = pd.DataFrame(
            [
                {
                    "date": d,
                    "contract": code,
                    "multiplier": 10.0,
                    "open": 100.0,
                    "spec_available_at": "2022-01-01T16:00:00+08:00",
                }
                for d in dates
                for code in codes
            ]
        )
        before = fixed_budget_orders(
            curves, pd.DataFrame({"date": dates}), bars, "carry", "2023-01-03"
        )
        bars.loc[bars.date.ge(dates[1]), "open"] *= 10
        after = fixed_budget_orders(
            curves, pd.DataFrame({"date": dates}), bars, "carry", "2023-01-03"
        )
        pd.testing.assert_frame_equal(before, after)
