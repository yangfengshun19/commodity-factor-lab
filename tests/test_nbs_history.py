import hashlib
import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from test_nbs import article

from commodity_factor_lab.nbs_archive import (
    HistoryCapture,
    freeze_history_catalog,
    official_links,
    period_from_title,
)
from commodity_factor_lab.nbs_history import run_nbs_history, series_identity_changes
from commodity_factor_lab.nbs_search import PublicNbsSearch, public_search_links

ROOT = Path(__file__).resolve().parents[1]


def fixture(root):
    (root / "configs").mkdir()
    raw = root / "raw"
    (raw / "discovery").mkdir(parents=True)
    (raw / "search_provider").mkdir()
    (root / "curves.csv").write_text("date,code\n2020-01-14,cu\n2020-01-14,al\n", encoding="utf-8")
    (root / "calendar.csv").write_text(
        "date,signal_cutoff\n2020-01-14,2020-01-14T12:30:00+00:00\n2020-02-04,2020-02-04T12:30:00+00:00\n2025-12-31,2025-12-31T12:30:00+00:00\n",
        encoding="utf-8",
    )
    (root / "prior.csv").write_text("analysis\na\nb\n", encoding="utf-8")
    text = 'registered_utc="2026-01-01T00:00:00+00:00"\n[paths]\ncurves="curves.csv"\ncalendar="calendar.csv"\nprior_exploration="prior.csv"\nseed="seed"\nraw="raw"\noutput="reports"\n[input_hashes]\n'
    for key, file in [
        ("curves", "curves.csv"),
        ("calendar", "calendar.csv"),
        ("prior_exploration", "prior.csv"),
    ]:
        text += f'{key}="{hashlib.sha256((root / file).read_bytes()).hexdigest()}"\n'
    base = (ROOT / "configs/nbs_history.toml").read_text(encoding="utf-8")
    text += base[base.index("[history]") :]
    config = root / "configs/history.toml"
    config.write_text(text, encoding="utf-8")
    cancellations = tomllib.loads(
        (ROOT / "configs/nbs_history_rules.toml").read_text(encoding="utf-8")
    )["cancellations"]
    records, rules_text = [], ""
    for year in sorted({int(item["period"][:4]) for item in cancellations}):
        items = [item for item in cancellations if item["period"].startswith(str(year))]
        content = ("<p>" + " ".join(item["evidence_fragment"] for item in items) + "</p>").encode()
        file = f"schedule_{year}.html"
        (raw / "discovery" / file).write_bytes(content)
        records.append(
            {
                "id": f"schedule_{year}",
                "url": items[0]["source_url"],
                "file": file,
                "sha256": hashlib.sha256(content).hexdigest(),
                "status": "http_ok",
            }
        )
    for item in cancellations:
        record = next(r for r in records if r["id"] == item["schedule_id"])
        rules_text += "[[cancellations]]\n" + "".join(
            f"{key}={json.dumps(record['sha256'] if key == 'source_sha256' else value, ensure_ascii=False)}\n"
            for key, value in item.items()
        )
    rules = root / "configs/rules.toml"
    rules.write_text(rules_text, encoding="utf-8")
    cancelled = {item["period"] for item in cancellations}
    observations = [
        {
            "period": f"{y}-{m:02d}-{part}",
            "url": f"https://www.stats.gov.cn/sj/zxfb/test/{y}-{m:02d}-{part}.html",
            "title": "synthetic test source",
        }
        for y in range(2020, 2026)
        for m in range(1, 13)
        for part in ["early", "middle", "late"]
        if f"{y}-{m:02d}-{part}" not in cancelled
    ]
    (raw / "discovered_links.json").write_text(
        json.dumps({"observations": observations}), encoding="utf-8"
    )
    (raw / "discovery/manifest.json").write_text(
        json.dumps(
            {
                "protocol_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "records": records,
                "requests": 0,
                "access_stop": False,
            }
        ),
        encoding="utf-8",
    )
    (raw / "search_provider/capture_manifest.json").write_text(
        json.dumps(
            {
                "protocol_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "records": [],
                "requests": 0,
                "access_stop": False,
            }
        ),
        encoding="utf-8",
    )
    (raw / "search_provider/manifest.json").write_text("[]", encoding="utf-8")
    (raw / "search_provider/site_config.json").write_text(
        json.dumps({"data": {"tabs": [{"tabName": "all", "maxPage": 10}]}}), encoding="utf-8"
    )
    return config, rules


class NbsHistoryTests(unittest.TestCase):
    def test_primary_link_filter_and_period_identity(self):
        content = '<a href="./2020.html">2020年1月上旬流通领域重要生产资料市场价格变动情况</a><a href="https://example.invalid/">other</a><a href="javascript:void(0)">x</a>'.encode()
        links = official_links(content, "https://www.stats.gov.cn/sj/zxfb/")
        self.assertEqual(len(links), 1)
        self.assertEqual(period_from_title(links[0]["title"]), "2020-01-early")
        self.assertIsNone(period_from_title("2020年13月上旬流通领域重要生产资料市场价格变动情况"))
        self.assertIsNone(period_from_title("unrelated report"))

    def test_complete_period_catalog_and_pin_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            config, rules = fixture(Path(directory))
            catalog = freeze_history_catalog(config, rules)
            self.assertEqual(len(catalog["expected_periods"]), 216)
            self.assertEqual(len(catalog["releases"]), 210)
            self.assertEqual(len(catalog["cancellations"]), 6)
            self.assertEqual(freeze_history_catalog(config, rules), catalog)
            path = Path(directory) / "raw/catalog.json"
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "Frozen catalog has changed"):
                freeze_history_catalog(config, rules)

    def test_missing_period_or_published_cancelled_period_prevents_freezing(self):
        for conflict in [False, True]:
            with tempfile.TemporaryDirectory() as directory:
                config, rules = fixture(Path(directory))
                path = Path(directory) / "raw/discovered_links.json"
                data = json.loads(path.read_text(encoding="utf-8"))
                if conflict:
                    data["observations"].append(
                        {"period": "2024-02-middle", "url": "https://www.stats.gov.cn/test.html"}
                    )
                else:
                    data["observations"].pop()
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    freeze_history_catalog(config, rules)

    def test_frozen_cancellation_requires_matching_primary_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            config, rules = fixture(Path(directory))
            target = next((Path(directory) / "raw/discovery").glob("*.html"))
            target.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Cancellation claim"):
                freeze_history_catalog(config, rules)

    def test_transport_retry_is_bounded_and_success_is_reused_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = fixture(Path(directory))
            connection = Mock()
            response = Mock(status=200)
            response.getheader.return_value = None
            response.read.side_effect = [b"source", b""]
            connection.getresponse.side_effect = [OSError("temporary transport"), response]
            with (
                patch(
                    "commodity_factor_lab.nbs_archive.http.client.HTTPSConnection",
                    return_value=connection,
                ),
                patch("commodity_factor_lab.nbs_archive.time.sleep"),
            ):
                capture = HistoryCapture(config, "releases")
                self.assertEqual(
                    capture.get("test", "https://www.stats.gov.cn/test.html", "release"), b"source"
                )
                self.assertEqual(capture.manifest["requests"], 2)
                self.assertEqual(
                    capture.get("test", "https://www.stats.gov.cn/test.html", "release"), b"source"
                )
                self.assertEqual(connection.request.call_count, 2)
                capture.close()

    def test_access_limit_stops_immediately_and_oversize_is_not_retried(self):
        for status, body in [(429, b"limited"), (200, b"x" * 3_000_001)]:
            with tempfile.TemporaryDirectory() as directory:
                config, _ = fixture(Path(directory))
                connection = Mock()
                response = Mock(status=status)
                response.getheader.return_value = None
                response.read.side_effect = [body, b""]
                connection.getresponse.return_value = response
                with (
                    patch(
                        "commodity_factor_lab.nbs_archive.http.client.HTTPSConnection",
                        return_value=connection,
                    ),
                    patch("commodity_factor_lab.nbs_archive.time.sleep"),
                ):
                    capture = HistoryCapture(config, "releases")
                    with self.assertRaises(ValueError):
                        capture.get("test", "https://www.stats.gov.cn/test.html", "release")
                    self.assertEqual(capture.manifest["requests"], 1)
                    if status == 429:
                        with self.assertRaises(ValueError):
                            HistoryCapture(config, "releases")

    def test_search_keeps_only_primary_links_and_never_ingests_summaries(self):
        title = "2020年1月上旬流通领域重要生产资料市场价格变动情况"
        result = {
            "resultDocs": [
                {
                    "data": {
                        "titleO": title,
                        "url": "https://www.stats.gov.cn/test.html",
                        "summary": "price=1234",
                    }
                },
                {"data": {"titleO": title, "url": "https://weibo.com/test"}},
            ]
        }
        links = public_search_links(result, "test")
        self.assertEqual(len(links), 1)
        self.assertNotIn("summary", links[0])
        self.assertNotIn("reference_price", links[0])

    def test_public_search_page_limit_and_429_are_not_bypassed(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = fixture(Path(directory))
            capture = PublicNbsSearch(config)
            with self.assertRaises(ValueError):
                capture.request("bad", "s", {"siteCode": "bm36000002", "page": 11})
            response = Mock(status=429)
            response.read.return_value = b"limited"
            connection = Mock()
            connection.getresponse.return_value = response
            with (
                patch(
                    "commodity_factor_lab.nbs_search.http.client.HTTPSConnection",
                    return_value=connection,
                ),
                patch("commodity_factor_lab.nbs_search.time.sleep"),
            ):
                with self.assertRaises(ValueError):
                    capture.request("one", "s", {"siteCode": "bm36000002", "page": 1})
            with self.assertRaises(ValueError):
                PublicNbsSearch(config)

    def test_full_offline_build_accounts_for_cancellations_and_blocks_source_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, rules = fixture(root)
            catalog = freeze_history_catalog(config, rules)
            folder = root / "raw/releases"
            folder.mkdir()
            records = []
            for item in catalog["releases"]:
                period = item["period"]
                year, month, part = period.split("-")
                year, month = int(year), int(month)
                if part == "late":
                    month += 1
                if month == 13:
                    year += 1
                    month = 1
                day = {"early": 14, "middle": 24, "late": 4}[part]
                content = article(period, f"{year}/{month:02d}/{day:02d} 09:30", copies=1)
                file = period + ".html"
                (folder / file).write_bytes(content)
                records.append(
                    {
                        "id": period,
                        "url": item["url"],
                        "kind": "release",
                        "file": file,
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "bytes": len(content),
                        "status": "http_ok",
                    }
                )
            manifest = {
                "protocol_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "catalog_sha256": hashlib.sha256(
                    (root / "raw/catalog.json").read_bytes()
                ).hexdigest(),
                "records": records,
                "requests": 210,
                "access_stop": False,
            }
            (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            output = run_nbs_history(config, rules)
            run = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertTrue(run["complete_normalized_history"])
            self.assertEqual(run["price_records"], 10500)
            self.assertEqual(run["officially_cancelled_periods"], 6)
            self.assertEqual(run["prediction_tests_added"], 0)
            self.assertFalse((output / "daily_ic.csv").exists())
            (folder / records[0]["file"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Article source identity"):
                run_nbs_history(config, rules)

    def test_metadata_breaks_do_not_treat_price_changes_as_new_identities(self):
        values = pd.DataFrame(
            [
                {
                    "related_code": "cu",
                    "period": "2020-01-early",
                    "period_start": pd.Timestamp("2020-01-01"),
                    "product": "铜",
                    "monitoring_scope": "24",
                    "specification_description": "same",
                    "comparability_key": "one",
                    "reference_price": 100,
                },
                {
                    "related_code": "cu",
                    "period": "2020-01-middle",
                    "period_start": pd.Timestamp("2020-01-11"),
                    "product": "铜",
                    "monitoring_scope": "24",
                    "specification_description": "same",
                    "comparability_key": "one",
                    "reference_price": 200,
                },
                {
                    "related_code": "cu",
                    "period": "2020-01-late",
                    "period_start": pd.Timestamp("2020-01-21"),
                    "product": "铜",
                    "monitoring_scope": "31",
                    "specification_description": "same",
                    "comparability_key": "two",
                    "reference_price": 300,
                },
            ]
        )
        changes = series_identity_changes(values)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes.iloc[0].previous_observed_period, "2020-01-middle")
        self.assertFalse(changes.iloc[0].exact_effective_date_certified)


if __name__ == "__main__":
    unittest.main()
