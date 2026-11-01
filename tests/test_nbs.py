import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from commodity_factor_lab.nbs import (
    align_release_events,
    map_industrial_references,
    parse_nbs_release,
    period_dates,
)
from commodity_factor_lab.nbs_ingest import capture_nbs, nbs_plan
from commodity_factor_lab.nbs_readiness import run_nbs_readiness

PRODUCTS = [
    ("电解铜", "1#"),
    ("铝锭", "A00"),
    ("锌锭", "0#"),
    ("铅锭", "1#"),
    ("螺纹钢", "Φ20mm，HRB400E"),
    ("热轧普通板卷", "4.75—11.5mm，Q235"),
    ("天然橡胶", "标准胶SCRWF"),
    ("纸浆", "进口针叶浆"),
] + [("示例品", f"型号{i}") for i in range(42)]


def article(
    period="2024-01-late", clock="2024/02/04 09:30", products=None, scope="全国31个省", copies=2
):
    products = products or PRODUCTS
    year, month, part = period.split("-")
    part = {"early": "上", "middle": "中", "late": "下"}[part]
    title = f"{year}年{int(month)}月{part}旬流通领域重要生产资料市场价格变动情况"

    def tab(rows):
        return (
            "<table>"
            + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
            + "</table>"
        )

    values = [["产品名称", "单位", "本期价格（元）", "比上期价格涨跌（元）", "涨跌幅（%）"]]
    specs = [["序号", "监测产品", "规格型号", "说明"]]
    for i, (name, grade) in enumerate(products):
        values.append([f"{name}（{grade}）", "吨", str(1000 + i), "7.1", "0.1"])
        specs.append([i + 1, name, grade, "明确的人工规格说明"])
    return (
        f"<title>{title} - 国家统计局</title><h1>{title}</h1><div>{clock}</div>据对{scope}监测。监测范围涵盖{scope}。"
        + (tab(values) + tab(specs)) * copies
    ).encode()


def fixture(root):
    (root / "configs").mkdir()
    (root / "raw").mkdir()
    (root / "curves.csv").write_text("date,code\n2024-02-05,cu\n2024-02-05,al\n", encoding="utf-8")
    (root / "calendar.csv").write_text(
        "date,signal_cutoff\n2024-02-02,2024-02-02 12:30:00+00:00\n2024-02-05,2024-02-05 12:30:00+00:00\n2025-12-31,2025-12-31 12:30:00+00:00\n",
        encoding="utf-8",
    )
    (root / "prior.csv").write_text("analysis\na\nb\n", encoding="utf-8")
    text = """registered_utc="2024-01-01T00:00:00+00:00"
[paths]
curves="curves.csv"
calendar="calendar.csv"
prior_exploration="prior.csv"
raw="raw"
output="reports"
[input_hashes]
"""
    for name, file in [
        ("curves", "curves.csv"),
        ("calendar", "calendar.csv"),
        ("prior_exploration", "prior.csv"),
    ]:
        text += f'{name}="{hashlib.sha256((root / file).read_bytes()).hexdigest()}"\n'
    text += """[study]
kind="industrial_reference_source_readiness_only"
prediction_tests_added=0
predictive_statistics_computed=false
minimum_assets_unchanged=8
allow_url_date_as_publication=false
allow_missing_release_inversion=false
strict_spot_basis_claim_eligible=false
published_reference_is_executable_spot=false
[collection]
minimum_interval_seconds=1.0
maximum_response_bytes=3000000
maximum_requests=11
stop_after_consecutive_failures=3
[negative_controls]
unpublished_periods=["2024-02-middle","2025-01-late"]
allow_reconstruction_from_later_change=false
"""
    sources = []
    for period in ("2024-01-late", "2025-12-late"):
        item = {
            "id": period,
            "period": period,
            "kind": "release",
            "url": "https://www.stats.gov.cn/test/" + period + ".html",
        }
        sources.append(item)
        text += (
            f'[[sources]]\nid="{period}"\nperiod="{period}"\nkind="release"\nurl="{item["url"]}"\n'
        )
    config = root / "configs/assay.toml"
    config.write_text(text, encoding="utf-8")
    records = []
    for i, item in enumerate(sources):
        content = article(item["period"], "2024/02/04 09:30" if i == 0 else "2026/01/04 09:30")
        file = item["id"] + ".html"
        (root / "raw" / file).write_bytes(content)
        records.append(
            {
                **item,
                "file": file,
                "sha256": hashlib.sha256(content).hexdigest(),
                "status": "http_ok",
            }
        )
    manifest = {
        "created_utc": "2026-10-30T00:00:00+00:00",
        "protocol_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "planned_sources": sources,
        "records": records,
    }
    (root / "raw/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return config


class NbsTests(unittest.TestCase):
    def parse(self, content=None):
        return parse_nbs_release(content or article(), "2024-01-late")

    def test_mobile_desktop_duplicates_deduplicate_without_doubling_observations(self):
        prices, specs, audit = self.parse()
        self.assertEqual(len(prices), 50)
        self.assertEqual(len(specs), 50)
        self.assertEqual(audit["identical_price_table_copies"], 2)
        self.assertEqual(
            prices.iloc[0].source_declared_published_at, pd.Timestamp("2024-02-04T01:30Z")
        )
        self.assertFalse(prices.executable_quote.any())
        self.assertFalse(prices.historical_first_vintage_certified.any())
        note = (
            article()
            .decode()
            .replace(
                "螺纹钢（Φ20mm，HRB400E）", "螺纹钢（Φ20mm，HRB400E）<sup>注</sup> <sup>2</sup>"
            )
        )
        noted, _, _ = self.parse(note.encode())
        self.assertFalse(noted["product"].str.contains("注").any())
        marked = article().replace("本期价格（元）".encode(), "本期价格<sup>1</sup>（元）".encode())
        self.assertEqual(self.parse(marked)[0].iloc[0].reference_price, 1000)

    def test_conflicting_copies_wrong_period_missing_clock_and_bad_number_rejected(self):
        bad_copy = article().decode().replace("1000", "9999", 1).encode()
        for content in [
            bad_copy,
            article(period="2024-02-early", clock="2024/02/14 09:30"),
            article(clock=""),
            article(clock="2024/01/30 09:30"),
            article().replace(b">1000<", b">1,00<"),
        ]:
            with self.subTest(content=content[:50]), self.assertRaises(ValueError):
                self.parse(content)
        fake_metadata = article(clock="").replace(
            b"<h1>", b'<meta name="CreationTime" content="2024/02/02 09:30"><h1>'
        )
        with self.assertRaises(ValueError):
            self.parse(fake_metadata)

    def test_scope_specification_and_product_form_do_not_silently_splice(self):
        prices, _, _ = self.parse()
        old_products = list(PRODUCTS)
        old_products[4] = ("螺纹钢", "Φ16-25mm，HRB400E")
        old_products[5] = ("热轧普通薄板", "3mm，Q235")
        old_products[7] = ("纸浆", "漂白化学浆")
        old, _, _ = self.parse(article(products=old_products, scope="24个省"))
        self.assertNotEqual(prices.iloc[0].comparability_key, old.iloc[0].comparability_key)
        mapped = map_industrial_references(old, {"cu", "al", "zn", "rb", "hc", "ru", "sp"})
        self.assertEqual(mapped.reference_mapping_usable.sum(), 4)
        self.assertFalse(mapped.strict_spot_basis_eligible.any())
        self.assertFalse(mapped.loc[mapped.related_code.eq("pb"), "in_existing_universe"].any())

    def test_unknown_scope_and_missing_specifications_are_not_usable(self):
        prices, _, _ = self.parse(article(scope="未说明地区"))
        result = map_industrial_references(prices, {"cu"})
        self.assertFalse(result.reference_mapping_usable.any())
        prices["monitoring_scope"] = "nationwide_31_provinces"
        prices["specification_description"] = ""
        self.assertFalse(map_industrial_references(prices, {"cu"}).reference_mapping_usable.any())

    def test_duplicate_product_missing_or_nonpositive_values_fail_closed(self):
        bad = list(PRODUCTS)
        bad[-1] = bad[0]
        with self.assertRaises(ValueError):
            self.parse(article(products=bad))
        with self.assertRaises(ValueError):
            self.parse(article().replace(b">1000<", b">0<"))
        prices, _, _ = self.parse(article().replace(b">1000<", b">-<"))
        self.assertTrue(pd.isna(prices.iloc[0].reference_price))
        self.assertFalse(map_industrial_references(prices, {"cu"}).reference_mapping_usable.any())

    def test_period_boundaries_include_actual_leap_month_end(self):
        start, end = period_dates("2024-02-late")
        self.assertEqual(str(start.date()), "2024-02-21")
        self.assertEqual(str(end.date()), "2024-02-29")
        with self.assertRaises(ValueError):
            period_dates("2024-13-late")

    def test_event_alignment_respects_holiday_exact_cutoff_and_one_nanosecond_future(self):
        prices, _, _ = self.parse()
        calendar = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-02-02", "2024-02-05", "2024-02-06"]),
                "signal_cutoff": pd.to_datetime(
                    ["2024-02-02T12:30Z", "2024-02-05T12:30Z", "2024-02-06T12:30Z"], utc=True
                ),
            }
        )
        event = align_release_events(prices, calendar).iloc[0]
        self.assertEqual(str(event.first_signal_date.date()), "2024-02-05")
        self.assertEqual(str(event.previous_signal_date.date()), "2024-02-02")
        prices["source_declared_published_at"] = calendar.iloc[1].signal_cutoff
        self.assertEqual(
            align_release_events(prices, calendar).iloc[0].first_signal_date, calendar.iloc[1].date
        )
        prices["source_declared_published_at"] += pd.Timedelta(1, unit="ns")
        self.assertEqual(
            align_release_events(prices, calendar).iloc[0].first_signal_date, calendar.iloc[2].date
        )

    def test_future_release_is_not_backfilled_and_naive_clock_rejected(self):
        prices, _, _ = self.parse()
        calendar = pd.DataFrame(
            {"date": pd.to_datetime(["2024-02-02"]), "signal_cutoff": ["2024-02-02T12:30Z"]}
        )
        event = align_release_events(prices, calendar).iloc[0]
        self.assertTrue(pd.isna(event.first_signal_date))
        self.assertEqual(event.status, "outside_frozen_calendar")
        prices["source_declared_published_at"] = pd.Timestamp("2024-02-04")
        with self.assertRaises(ValueError):
            align_release_events(prices, calendar)

    def test_offline_assay_retains_missing_periods_and_never_creates_daily_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory))
            output = run_nbs_readiness(config)
            result = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(result["price_records"], 100)
            self.assertEqual(result["source_clock_events_in_calendar"], 1)
            self.assertFalse(result["predictive_statistics_computed"])
            self.assertEqual(result["reconstructed_missing_periods"], 0)
            self.assertFalse((output / "daily_ic.csv").exists())
            data = pd.read_csv(output / "published_reference_prices.csv")
            self.assertEqual(set(data.period), {"2024-01-late", "2025-12-late"})

    def test_protocol_rejects_clock_shortcut_external_host_and_changed_frozen_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            original = config.read_text(encoding="utf-8")
            for changed in [
                original.replace(
                    "allow_url_date_as_publication=false", "allow_url_date_as_publication=true"
                ),
                original.replace("www.stats.gov.cn", "example.invalid"),
            ]:
                config.write_text(changed, encoding="utf-8")
                with self.assertRaises(ValueError):
                    nbs_plan(config)
            config.write_text(original, encoding="utf-8")
            (root / "curves.csv").write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                nbs_plan(config)

    def test_access_limit_stops_capture_and_blocks_an_unchanged_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            (root / "raw/manifest.json").unlink()
            # Unregistered synthetic files would be rejected; use fresh names for this capture.
            text = config.read_text(encoding="utf-8").replace('raw="raw"', 'raw="capture"')
            config.write_text(text, encoding="utf-8")
            response = Mock(status=429)
            response.read.return_value = b""
            response.getheader.return_value = None
            connection = Mock()
            connection.getresponse.return_value = response
            with (
                patch(
                    "commodity_factor_lab.nbs_ingest.http.client.HTTPSConnection",
                    return_value=connection,
                ),
                patch("commodity_factor_lab.nbs_ingest.time.sleep"),
            ):
                result = capture_nbs(config)
                self.assertEqual(len(result["records"]), 1)
                with self.assertRaisesRegex(ValueError, "Previous access/failure stop"):
                    capture_nbs(config)
            self.assertEqual(connection.request.call_count, 1)

    def test_independent_raw_reader_catches_a_deliberately_wrong_price_with_bom_calendar(self):
        source = Path(__file__).resolve().parents[1] / "scripts/reconcile_nbs_probe.py"
        spec = importlib.util.spec_from_file_location("independent_nbs_check", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            output = run_nbs_readiness(config)
            path = output / "published_reference_prices.csv"
            values = pd.read_csv(path)
            values.loc[0, "reference_price"] += 100
            values.to_csv(path, index=False)
            calendar = root / "calendar.csv"
            calendar.write_bytes(b"\xef\xbb\xbf" + calendar.read_bytes())
            result = module.reconcile(root / "raw", output, calendar, root / "audit")
            self.assertEqual(result["differences"], 1)
            checks = pd.read_csv(root / "audit/checks.csv")
            self.assertEqual(checks.loc[~checks.passed, "check"].tolist(), ["price_matches_raw"])


if __name__ == "__main__":
    unittest.main()
