import unittest

import numpy as np
import pandas as pd

from commodity_factor_lab.spot import match_spot_diagnostics, parse_spot_page


def page(rows, dated=True, heading="上海黄金交易所2024年1月30日交易行情"):
    header = ["合约", "开盘价", "最高价", "最低价", "收盘价", "成交量（kg）", "成交金额（元）"]
    if dated:
        header.insert(0, "日期")
    table = "<tr>" + "".join(f"<th>{cell}</th>" for cell in header) + "</tr>"
    for row in rows:
        table += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    # A commented-out table cell must not shift actual columns.
    return f"<h1>{heading}</h1><table><!-- <td>序号</td> -->{table}</table>".encode()


def row(symbol="Au99.95", close="500", volume="14", day="2024-01-30"):
    return [day, symbol, "500", "505", "495", close, volume, "7000000"]


class SpotTests(unittest.TestCase):
    def parse(self, rows):
        return parse_spot_page(page(rows), "2024-01-30", "dated_query")[0]

    def curves(self):
        return pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-30")] * 2,
                "code": ["au", "ag"],
                "front_contract": ["au2404", "ag2404"],
                "reference_settle": [510.0, 6000.0],
                "status": ["ok", "ok"],
            }
        )

    def test_quote_units_are_not_lot_values_and_gold_is_one_underlying(self):
        quotes = self.parse([row(), row("Au99.99"), row("Ag99.99")])
        quotes.loc[quotes.code.eq("ag"), ["open", "high", "low", "close"]] = [
            6000,
            6000,
            6000,
            6000,
        ]
        matched = match_spot_diagnostics(quotes, self.curves())
        self.assertEqual(matched.loc[matched.diagnostic_available, "code"].nunique(), 2)
        self.assertAlmostEqual(matched.iloc[0].diagnostic_log_quote_ratio, np.log(500 / 510))
        self.assertTrue(matched.quote_multiplier.eq(1).all())
        self.assertTrue(matched.strict_spot_basis.isna().all())
        self.assertFalse(matched.claim_eligible.any())
        self.assertFalse(matched.publisher_clock_verified.any())
        self.assertEqual(matched.iloc[2].kind, "physical_spot_t2")

    def test_wrong_date_and_duplicate_instrument_are_rejected(self):
        for rows in ([row(day="2024-01-31")], [row(), row()]):
            with self.assertRaises(ValueError):
                self.parse(rows)

    def test_historical_body_heading_must_match_requested_date(self):
        content = page([row()[1:]], dated=False)
        result, audit = parse_spot_page(content, "2024-01-30", "historical_article")
        self.assertEqual(result.iloc[0].close, 500)
        self.assertEqual(audit["date_identity"], "article_heading")
        with self.assertRaises(ValueError):
            parse_spot_page(content, "2024-01-31", "historical_article")
        with self.assertRaises(ValueError):
            parse_spot_page(page([], heading="Latest prices"), "2024-01-30", "historical_article")

    def test_empty_holiday_table_preserves_missing(self):
        result, audit = parse_spot_page(page([]), "2024-02-12", "dated_query")
        self.assertTrue(result.close.isna().all())
        self.assertTrue(result.quote_status.eq("empty_daily_table").all())
        self.assertEqual(audit["table_rows"], 0)
        self.assertFalse(match_spot_diagnostics(result, self.curves()).diagnostic_available.any())

    def test_zero_volume_cannot_turn_stale_close_into_a_signal(self):
        quotes = self.parse([row(volume="0")])
        self.assertEqual(quotes.iloc[0].quote_status, "no_trade")
        self.assertTrue(
            match_spot_diagnostics(quotes, self.curves()).diagnostic_log_quote_ratio.isna().all()
        )

    def test_missing_invalid_prices_and_bad_range_stay_unusable(self):
        for close, expected in [
            ("-", "missing_or_invalid_ohlc"),
            ("900", "invalid_ohlc_range"),
            ("0", "missing_or_invalid_ohlc"),
        ]:
            quotes = self.parse([row(close=close)])
            self.assertEqual(quotes.iloc[0].quote_status, expected)
            self.assertFalse(
                match_spot_diagnostics(quotes, self.curves()).diagnostic_available.any()
            )
        with self.assertRaises(ValueError):
            self.parse([row(close="NaN")])
        with self.assertRaises(ValueError):
            self.parse([row(close="1,00")])

    def test_deferred_and_international_gold_do_not_replace_physical_spot(self):
        result, audit = parse_spot_page(
            page([row(s) for s in ["Au(T+D)", "Ag(T+D)", "mAu(T+D)", "iAu99.99"]]),
            "2024-01-30",
            "dated_query",
        )
        self.assertEqual(audit["rejected_control_instruments_present"], 4)
        self.assertTrue(result.quote_status.eq("symbol_absent").all())
        wrong = self.parse([row()])
        wrong.loc[0, "symbol"] = "Au(T+D)"
        with self.assertRaises(ValueError):
            match_spot_diagnostics(wrong, self.curves())

    def test_matching_is_exact_date_selected_contract_and_explicit_unit(self):
        quotes = self.parse([row()])
        curves = self.curves()
        curves["date"] += pd.Timedelta(1, unit="d")
        self.assertFalse(match_spot_diagnostics(quotes, curves).diagnostic_available.any())
        curves = self.curves()
        curves.loc[0, "front_contract"] = "au0"
        with self.assertRaises(ValueError):
            match_spot_diagnostics(quotes, curves)
        quotes.loc[0, "quote_unit"] = "CNY/kg"
        with self.assertRaises(ValueError):
            match_spot_diagnostics(quotes, self.curves())

    def test_invalid_curve_price_and_wrong_contract_code_are_rejected(self):
        curves = self.curves()
        curves.loc[0, "reference_settle"] = 0
        self.assertFalse(
            match_spot_diagnostics(self.parse([row()]), curves).diagnostic_available.any()
        )
        curves.loc[0, "front_contract"] = "ag2404"
        with self.assertRaises(ValueError):
            match_spot_diagnostics(self.parse([row()]), curves)

    def test_gold_policy_tag_never_applies_an_automatic_tax_adjustment(self):
        quotes = self.parse([row()])
        curves = self.curves()
        quotes["date"] = pd.Timestamp("2025-11-03")
        curves["date"] = pd.Timestamp("2025-11-03")
        matched = match_spot_diagnostics(quotes, curves)
        self.assertEqual(matched.iloc[0].gold_withdrawal_policy_regime, "post_2025_11_01_unmapped")
        self.assertAlmostEqual(matched.iloc[0].diagnostic_log_quote_ratio, np.log(500 / 510))
        self.assertFalse(matched.brand_location_tax_history_verified.any())


if __name__ == "__main__":
    unittest.main()
