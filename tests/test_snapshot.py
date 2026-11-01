import json
import unittest

from commodity_factor_lab.snapshot import parse_sina_daily


class SnapshotParsingTests(unittest.TestCase):
    def test_provider_jsonp_is_parsed_as_data_without_executing_javascript(self):
        record = {"d": "2023-01-03", "o": "100", "h": "102", "l": "99", "c": "101", "v": "1234"}
        payload = ("/* banner */ var _data=(" + json.dumps([record]) + ");").encode()
        frame = parse_sina_daily(payload)
        self.assertEqual(frame.iloc[0]["open"], 100)
        self.assertEqual(frame.iloc[0]["volume"], 1234)

    def test_non_data_response_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_sina_daily(b"<html>Service unavailable</html>")

    def test_duplicate_provider_dates_are_not_silently_accepted(self):
        record = {"d": "2023-01-03", "o": "100", "h": "102", "l": "99", "c": "101", "v": "1234"}
        with self.assertRaises(ValueError):
            parse_sina_daily(("var _data=(" + json.dumps([record, record]) + ");").encode())
