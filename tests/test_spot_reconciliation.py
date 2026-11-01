import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


class SpotReconciliationTests(unittest.TestCase):
    def test_independent_reader_detects_a_deliberately_wrong_ratio(self):
        source = Path(__file__).resolve().parents[1] / "scripts/reconcile_spot_probe.py"
        spec = importlib.util.spec_from_file_location("spot_check", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"records": []}), encoding="utf-8")
            # Empty captures cannot prove anything: use a complete synthetic raw table.
            header = [
                "日期",
                "合约",
                "开盘价",
                "最高价",
                "最低价",
                "收盘价",
                "涨跌",
                "涨幅",
                "加权平均价",
                "成交量",
                "成交金额",
                "市场持仓",
                "交收方向",
                "交收量",
            ]
            cells = [
                "2024-01-30",
                "Au99.95",
                "500",
                "505",
                "495",
                "500",
                "0",
                "0",
                "500",
                "14",
                "7000000",
                "-",
                "",
                "",
            ]
            html = (
                "<table><tr>"
                + "".join(f"<td>{h}</td>" for h in header)
                + "</tr><tr>"
                + "".join(f"<td>{c}</td>" for c in cells)
                + "</tr></table>"
            )
            content = html.encode()
            (root / "raw.html").write_bytes(content)
            import hashlib

            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "id": "2024-01-30",
                                "date": "2024-01-30",
                                "mode": "dated_query",
                                "file": "raw.html",
                                "sha256": hashlib.sha256(content).hexdigest(),
                                "status": "http_ok",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (root / "bars.csv").write_text(
                "date,contract,settle\n2024-01-30,au2404,500\n", encoding="utf-8"
            )
            fields = [
                "date",
                "symbol",
                "close",
                "volume_as_published",
                "strict_spot_basis",
                "claim_eligible",
                "publisher_available_at",
                "publisher_clock_verified",
                "diagnostic_available",
                "front_contract",
                "reference_settle",
                "diagnostic_log_quote_ratio",
            ]
            with (root / "quote_diagnostics.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for symbol in ("Au99.95", "Au99.99", "Ag99.99"):
                    row = {field: "" for field in fields}
                    row.update(
                        date="2024-01-30",
                        symbol=symbol,
                        claim_eligible="False",
                        publisher_clock_verified="False",
                        diagnostic_available="False",
                    )
                    if symbol == "Au99.95":
                        row.update(
                            close="500",
                            volume_as_published="14",
                            diagnostic_available="True",
                            front_contract="au2404",
                            reference_settle="500",
                            diagnostic_log_quote_ratio="7",
                        )
                    writer.writerow(row)
            result = module.reconcile(root, root, root / "bars.csv", root / "audit")
            self.assertEqual(result["differences"], 1)
            with (root / "audit/checks.csv").open(encoding="utf-8", newline="") as stream:
                failed = [
                    row["check"] for row in csv.DictReader(stream) if row["passed"] == "False"
                ]
            self.assertEqual(failed, ["log_ratio_independent_arithmetic"])


if __name__ == "__main__":
    unittest.main()
