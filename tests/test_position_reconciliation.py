import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from commodity_factor_lab.position_reconciliation import raw_totals, reconcile


class IndependentPositionAuditTests(unittest.TestCase):
    def payload(self, date, step=0):
        return {
            "report_date": date,
            "update_date": date + " 18:00:00",
            "o_cursor": [
                {
                    "INSTRUMENTID": "cu2405",
                    "RANK": i,
                    "PARTICIPANTID2": str(i),
                    "CJ2": 100 + i + step,
                    "PARTICIPANTID3": str(i + 30),
                    "CJ3": 50 + i,
                }
                for i in range(1, 21)
            ],
        }

    def test_invalid_members_or_duplicate_ranks_do_not_form_a_sum(self):
        payload = self.payload("20240130")
        self.assertEqual(raw_totals(payload, "20240130")["cu2405"][:2], (2210, 1210))
        payload["o_cursor"][0]["PARTICIPANTID2"] = payload["o_cursor"][1]["PARTICIPANTID2"]
        self.assertEqual(raw_totals(payload, "20240130"), {})
        payload = self.payload("20240130")
        payload["o_cursor"].append(payload["o_cursor"][-1])
        self.assertEqual(raw_totals(payload, "20240130"), {})

    def test_audit_detects_a_deliberately_wrong_flow(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, report, daily = root / "raw", root / "report", root / "daily"
            (raw / "responses").mkdir(parents=True)
            report.mkdir()
            daily.mkdir()
            dates = pd.bdate_range("2024-01-02", periods=25)
            text_dates = dates.strftime("%Y%m%d").tolist()
            records, bars, qualified, panel, calendar = [], [], [], [], []
            for step, date in enumerate(dates):
                text = text_dates[step]
                source = raw / "responses" / f"{text}.json"
                source.write_text(json.dumps(self.payload(text, step)), encoding="utf-8")
                records.append(
                    {
                        "date": text,
                        "status": "ok",
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                )
                iso = date.strftime("%Y-%m-%d")
                bars.append({"date": iso, "contract": "cu2405", "open_interest": 10000})
                qualified.append(
                    {
                        "date": iso,
                        "contract": "cu2405",
                        "long_top20": 2210 + 20 * step,
                        "short_top20": 1210,
                        "net_top20": 1000 + 20 * step,
                    }
                )
                panel.append(
                    {
                        "date": iso,
                        "contract": "cu2405",
                        "normalized_position_flow": np.nan
                        if step < 20
                        else 0.14
                        if step == 24
                        else 0.04,
                    }
                )
                calendar.append({"date": iso, "signal_cutoff": iso + "T20:30:00+08:00"})
            (raw / "manifest.json").write_text(
                json.dumps({"planned_dates": text_dates, "records": records}), encoding="utf-8"
            )
            (report / "run.json").write_text('{"synthetic":true}', encoding="utf-8")
            pd.DataFrame(qualified).to_csv(report / "qualified_contract_ranks.csv", index=False)
            pd.DataFrame(panel).to_csv(report / "selected_position_panel.csv", index=False)
            pd.DataFrame(bars).to_csv(root / "bars.csv", index=False)
            pd.DataFrame(calendar).to_csv(root / "calendar.csv", index=False)
            result = reconcile(
                raw, report, root / "bars.csv", root / "calendar.csv", daily, root / "audit"
            )
            self.assertEqual(result["checks"], 100)
            self.assertEqual(result["differences"], 1)
            errors = pd.read_csv(root / "audit/differences.csv")
            self.assertEqual(errors.iloc[0].kind, "same_contract_twenty_session_flow")
            self.assertAlmostEqual(errors.iloc[0].expected, 0.04)
            self.assertAlmostEqual(errors.iloc[0].observed, 0.14)
