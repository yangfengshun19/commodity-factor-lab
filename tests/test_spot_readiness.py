import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from commodity_factor_lab.spot_readiness import run_spot_readiness


class SpotReadinessTests(unittest.TestCase):
    def fixture(self, root):
        configs = root / "configs"
        configs.mkdir()
        raw = root / "raw"
        raw.mkdir()
        (root / "curves.csv").write_text(
            "date,code,front_contract,reference_settle,status\n2024-01-30,au,au2404,510,ok\n",
            encoding="utf-8",
        )
        (root / "calendar.csv").write_text("date\n2024-01-30\n", encoding="utf-8")
        (root / "prior.csv").write_text("analysis\na\nb\n", encoding="utf-8")
        protocol = """registered_utc="2024-01-01T00:00:00+00:00"
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
            protocol += f'{name}="{hashlib.sha256((root / file).read_bytes()).hexdigest()}"\n'
        protocol += """[probe]
dates=["2024-01-30","2024-02-12"]
holiday_negative_control="2024-02-12"
[study]
kind="data_readiness_only"
prediction_tests_added=0
predictive_statistics_computed=false
spot_basis_claim_eligible=false
require_positive_volume=true
require_publisher_clock=true
allow_assumed_publication_lag=false
require_brand_location_tax_mapping=true
min_inference_assets_unchanged=8
spot_price_field="close"
gold_tax_effective_date="2025-11-01"
"""
        for symbol, code, kind, unit in [
            ("Au99.95", "au", "physical_cash_t0", "CNY/g"),
            ("Au99.99", "au", "physical_cash_t0", "CNY/g"),
            ("Ag99.99", "ag", "physical_spot_t2", "CNY/kg"),
        ]:
            protocol += f'''[[instruments]]
symbol="{symbol}"
code="{code}"
kind="{kind}"
quote_unit="{unit}"
quote_multiplier=1.0
brand_location_tax_history_verified=false
'''
        config = configs / "assay.toml"
        config.write_text(protocol, encoding="utf-8")
        sources = configs / "sources.toml"
        sources.write_text("# offline synthetic catalog\n", encoding="utf-8")
        requests, records = [], []
        for date in ["2024-01-30", "2024-02-12"]:
            request = {
                "id": date,
                "date": date,
                "mode": "dated_query",
                "url": "https://example.invalid/" + date,
            }
            requests.append(request)
            content = (
                "<table><tr>"
                + "".join(
                    f"<th>{h}</th>"
                    for h in [
                        "日期",
                        "合约",
                        "开盘价",
                        "最高价",
                        "最低价",
                        "收盘价",
                        "成交量",
                        "成交金额",
                    ]
                )
                + "</tr>"
            )
            if date == "2024-01-30":
                content += (
                    "<tr>"
                    + "".join(
                        f"<td>{cell}</td>"
                        for cell in [date, "Au99.95", "500", "505", "495", "500", "14", "7000000"]
                    )
                    + "</tr>"
                )
            content += "</table>"
            file = date + ".html"
            (raw / file).write_text(content, encoding="utf-8")
            records.append(
                {
                    **request,
                    "file": file,
                    "sha256": hashlib.sha256((raw / file).read_bytes()).hexdigest(),
                    "status": "http_ok",
                }
            )
        manifest = {
            "protocol_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "sources_sha256": hashlib.sha256(sources.read_bytes()).hexdigest(),
            "created_utc": "2024-03-01T00:00:00+00:00",
            "planned_requests": requests,
            "records": records,
        }
        (raw / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return config, sources

    def test_offline_mixed_traded_and_all_missing_pages_have_no_prediction_output(self):
        with tempfile.TemporaryDirectory() as directory:
            config, sources = self.fixture(Path(directory))
            output = run_spot_readiness(config, sources)
            result = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(result["same_date_contract_diagnostics"], 1)
            self.assertEqual(result["strict_factor_observations"], 0)
            self.assertEqual(result["registered_exploration_family_size_unchanged"], 2)
            self.assertFalse(result["predictive_statistics_computed"])
            self.assertFalse((output / "daily_ic.csv").exists())
            self.assertTrue((output / "index.html").exists())

    def test_changed_input_and_clock_relaxation_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, sources = self.fixture(root)
            original = (root / "curves.csv").read_bytes()
            (root / "curves.csv").write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "Frozen assay input changed"):
                run_spot_readiness(config, sources)
            (root / "curves.csv").write_bytes(original)
            config.write_text(
                config.read_text(encoding="utf-8").replace(
                    "allow_assumed_publication_lag=false", "allow_assumed_publication_lag=true"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "cannot activate"):
                run_spot_readiness(config, sources)


if __name__ == "__main__":
    unittest.main()
