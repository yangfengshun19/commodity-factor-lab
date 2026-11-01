"""Hand-computed roll, reversal, variance and drawdown accounting examples."""

import unittest

import numpy as np
import pandas as pd

from commodity_factor_lab.market_attribution import (
    attribute_book,
    drawdown_attribution,
    risk_decomposition,
    target_diagnostics,
)


def accounting_fixture():
    days = pd.date_range("2023-01-03", periods=3)
    prices = [
        (0, "a1", "a", 100, 110),
        (0, "b1", "b", 50, 45),
        (1, "a1", "a", 112, 114),
        (1, "a2", "a", 200, 202),
        (1, "b1", "b", 46, 48),
        (2, "a2", "a", 201, 205),
        (2, "b1", "b", 47, 45),
    ]
    bars = pd.DataFrame(
        [
            {"date": days[d], "contract": c, "code": k, "open": o, "settle": s}
            for d, c, k, o, s in prices
        ]
    )
    events = [
        (0, "a", "a1", 0, 2, 2, 0, 2, 100, 1.0),
        (0, "b", "b1", 0, -3, -3, 0, 3, 50, 0.15),
        (1, "a", "a1", 2, 0, -2, 2, 0, 112, 1.12),
        (1, "a", "a2", 0, 1, 1, 0, 1, 200, 1.0),
        (1, "b", "b1", -3, 1, 4, 3, 1, 46, 0.184),
        (2, "a", "a2", 1, 0, -1, 1, 0, 201, 1.005),
        (2, "b", "b1", 1, 0, -1, 1, 0, 47, 0.047),
    ]
    trades = pd.DataFrame(
        events,
        columns=[
            "day",
            "code",
            "contract",
            "old_lots",
            "new_lots",
            "delta_lots",
            "closed_lots",
            "opened_lots",
            "fill_price",
            "fee",
        ],
    )
    trades["date"] = trades.day.map(dict(enumerate(days)))
    trades["reason"] = np.where(trades.day.eq(2), "terminal_liquidation", "rebalance")
    trades["notional"] = (
        trades.delta_lots.abs() * trades.fill_price * trades.code.map({"a": 10.0, "b": 2.0})
    )
    holdings = pd.DataFrame(
        [
            [days[0], "a1", "a", 2, 110, 10.0],
            [days[0], "b1", "b", -3, 45, 2.0],
            [days[1], "a2", "a", 1, 202, 10.0],
            [days[1], "b1", "b", 1, 48, 2.0],
        ],
        columns=["date", "contract", "code", "lots", "settle", "multiplier"],
    )
    daily = pd.DataFrame(
        {
            "date": days,
            "previous_equity": [10000.0, 10228.85, 10284.546],
            "equity": [10228.85, 10284.546, 10271.494],
            "overnight_pnl": [0.0, 34.0, -12.0],
            "intraday_pnl": [230.0, 24.0, 0.0],
            "fees": [1.15, 2.304, 1.052],
            "gross_notional": [2470.0, 2116.0, 0.0],
            "net_notional": [1930.0, 2116.0, 0.0],
        }
    )
    return bars, daily, trades, holdings


class MarketAttributionTests(unittest.TestCase):
    def test_roll_does_not_receive_a_cross_contract_price_jump(self):
        args = accounting_fixture()
        parts, checks = attribute_book(*args, ["a", "b"], {"a": 10.0, "b": 2.0}, 5.0, "manual")
        roll = parts.loc[parts.date.eq(args[1].date.iloc[1]) & parts.code.eq("a")].iloc[0]
        self.assertEqual(roll.gross_pnl, 60.0)
        self.assertEqual(roll.event, "roll_including_any_resize")
        self.assertAlmostEqual(parts.net_pnl.sum(), 271.494)
        self.assertAlmostEqual(parts.loc[parts.code.eq("a"), "net_pnl"].sum(), 245.875)
        self.assertLess(checks.filter(like="difference").abs().max().max(), 1e-8)

    def test_short_to_long_reversal_and_terminal_fee(self):
        args = accounting_fixture()
        parts, _ = attribute_book(*args, ["a", "b"], {"a": 10.0, "b": 2.0}, 5.0, "manual")
        b = parts.loc[parts.code.eq("b")].reset_index()
        self.assertAlmostEqual(b.iloc[1].opened_fee, 0.046)
        self.assertAlmostEqual(b.iloc[1].closed_fee, 0.138)
        self.assertEqual(b.iloc[1].event, "same_contract_resize_or_reverse")
        self.assertEqual(b.iloc[2].event, "terminal_liquidation")

    def test_inconsistent_holdings_and_modified_fee_are_detected(self):
        bars, daily, trades, holdings = accounting_fixture()
        bad = holdings.copy()
        bad.loc[0, "lots"] = 3
        with self.assertRaisesRegex(ValueError, "closing holdings"):
            attribute_book(
                bars, daily, trades, bad, ["a", "b"], {"a": 10.0, "b": 2.0}, 5.0, "manual"
            )
        bad = trades.copy()
        bad.loc[0, "fee"] += 1
        with self.assertRaisesRegex(ValueError, "split fee"):
            attribute_book(
                bars, daily, bad, holdings, ["a", "b"], {"a": 10.0, "b": 2.0}, 5.0, "manual"
            )

    def test_variance_contributions_can_be_negative_and_sum_to_one(self):
        rows = []
        for day, x in zip(pd.date_range("2023-01-01", periods=3), [-2.0, 0.0, 2.0], strict=True):
            for code, y in [("a", x), ("b", -x / 2)]:
                rows.append(
                    {
                        "date": day,
                        "code": code,
                        "net_pnl": y,
                        "gross_pnl": y,
                        "signed_notional": 0.0,
                        "gross_notional": 0.0,
                    }
                )
        table, risk = risk_decomposition(pd.DataFrame(rows), 1000.0)
        shares = table.set_index("code").variance_share
        self.assertAlmostEqual(shares["a"], 2.0)
        self.assertAlmostEqual(shares["b"], -1.0)
        self.assertAlmostEqual(shares.sum(), 1.0)
        self.assertAlmostEqual(risk["cash_normalized_net_volatility"], np.sqrt(252) * 0.001)

    def test_zero_portfolio_variance_has_no_fake_risk_shares(self):
        rows = []
        for day, x in zip(pd.date_range("2023-01-01", periods=3), [-2.0, 0.0, 2.0], strict=True):
            for code, y in [("a", x), ("b", -x)]:
                rows.append(
                    {
                        "date": day,
                        "code": code,
                        "net_pnl": y,
                        "gross_pnl": y,
                        "signed_notional": 0.0,
                        "gross_notional": 0.0,
                    }
                )
        table, risk = risk_decomposition(pd.DataFrame(rows), 1000.0)
        self.assertTrue(table.variance_share.isna().all())
        self.assertEqual(risk["cash_normalized_net_volatility"], 0.0)

    def test_drawdown_uses_peak_equity_and_excludes_peak_day_profit(self):
        days = pd.date_range("2023-01-03", periods=3)
        daily = pd.DataFrame({"date": days, "equity": [110.0, 90.0, 95.0]})
        rows = []
        for day, a, b in zip(days, [6.0, -12.0, 3.0], [4.0, -8.0, 2.0], strict=True):
            rows.extend(
                [{"date": day, "code": "a", "net_pnl": a}, {"date": day, "code": "b", "net_pnl": b}]
            )
        parts, window = drawdown_attribution(pd.DataFrame(rows), daily, 100.0, "2023-01-02")
        self.assertEqual(window["peak_date"], "2023-01-03")
        self.assertEqual(window["trough_date"], "2023-01-04")
        self.assertAlmostEqual(parts.drawdown_contribution.sum(), -20 / 110)
        self.assertEqual(window["window_sessions"], 1)

    def test_drawdown_includes_initial_cash_before_first_loss(self):
        day = pd.Timestamp("2023-01-03")
        parts = pd.DataFrame({"date": [day], "code": ["a"], "net_pnl": [-10.0]})
        _, window = drawdown_attribution(
            parts, pd.DataFrame({"date": [day], "equity": [90.0]}), 100.0, "2023-01-02"
        )
        self.assertEqual(window["peak_date"], "2023-01-02")
        self.assertAlmostEqual(window["max_drawdown"], -0.1)

    def test_rank_cancellation_can_concentrate_original_targets(self):
        day = pd.Timestamp("2023-01-03")
        signal = pd.Timestamp("2023-01-02")
        targets = []
        codes = ["a", "b", "c", "d"]
        arrays = {
            "momentum": [-0.375, -0.125, 0.125, 0.375],
            "momentum_carry": [0.0, -0.5, 0.5, 0.0],
            "equal_weight_long": [0.25] * 4,
        }
        for book, w in arrays.items():
            for code, weight in zip(codes, w, strict=True):
                targets.append(
                    {
                        "date": day,
                        "signal_date": signal,
                        "code": code,
                        "book": book,
                        "contract": code + "1",
                        "weight": weight,
                    }
                )
        curves = pd.DataFrame(
            {
                "date": [signal] * 4,
                "code": codes,
                "momentum_126d": [1, 2, 3, 4],
                "carry": [4, 2, 3, 1],
            }
        )
        structure, directions = target_diagnostics(pd.DataFrame(targets), curves)
        structure = structure.set_index("book")
        self.assertAlmostEqual(structure.loc["momentum_carry", "score_normalization_ratio"], 4.0)
        self.assertAlmostEqual(structure.loc["momentum_carry", "effective_absolute_names"], 2.0)
        self.assertGreater(
            structure.loc["momentum_carry", "max_absolute_weight"],
            structure.loc["momentum", "max_absolute_weight"],
        )
        self.assertEqual(int(directions.direction_state.eq("momentum_only_nonzero").sum()), 2)


if __name__ == "__main__":
    unittest.main()
