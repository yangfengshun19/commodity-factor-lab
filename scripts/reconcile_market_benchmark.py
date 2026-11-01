"""Independent target and market cash audit; imports no production research module."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Audit:
    def __init__(self):
        self.checks = 0
        self.differences = []

    def same(self, label, actual, expected):
        self.checks += 1
        if isinstance(actual, (float, np.floating)) or isinstance(expected, (float, np.floating)):
            ok = np.isclose(actual, expected, rtol=1e-11, atol=1e-6, equal_nan=True)
        else:
            ok = actual == expected
        if not ok:
            self.differences.append(f"{label}: {actual!r} != {expected!r}")


def centered_rank(values):
    """Average ties, implemented without the production rank helper."""
    numbers = np.asarray(values, dtype=float)
    ranks = np.empty(len(numbers))
    for value in np.unique(numbers):
        lower = int((numbers < value).sum())
        count = int((numbers == value).sum())
        ranks[numbers == value] = lower + (count + 1) / 2
    return ranks - ranks.mean()


def verify_targets(report, run, curves, calendar, audit):
    s = run["protocol"]["study"]
    multipliers = run["protocol"]["multipliers"]
    dates = pd.DatetimeIndex(calendar.date)
    evaluation = dates[(dates >= s["start"]) & (dates <= s["end"])]
    rows = {book: [] for book in s["books"]}
    for number, day in enumerate(evaluation):
        if number % 5 or day == evaluation[-1]:
            continue
        previous = dates[dates.get_loc(day) - 1]
        group = curves.loc[curves.date.eq(previous) & curves.code.isin(s["codes"])].copy()
        group = group.loc[
            np.isfinite(group[["momentum_126d", "carry", "reference_settle"]]).all(axis=1)
            & group.reference_settle.gt(0)
            & group.front_contract.notna()
            & group.front_expiry.notna()
        ].sort_values("code")
        # The observed frozen sample has ten common products on every decision.
        # A changed/gated sample must be separately audited, never silently skipped.
        audit.same(f"common universe {day}", len(group), len(s["codes"]))
        if len(group) < s["min_assets"]:
            raise ValueError("Audit requires explicit handling of changed common eligibility")
        momentum = centered_rank(group.momentum_126d)
        carry = centered_rank(group.carry)
        cutoff = calendar.loc[calendar.date.eq(previous), "signal_cutoff"].iloc[0]
        for book in s["books"]:
            if book == "equal_weight_long":
                allocation = np.ones(len(group)) / len(group)
            else:
                score = momentum if book == "momentum" else (momentum + carry) / 2
                gross = abs(score).sum()
                allocation = score / gross if gross else score * 0
            for weight, row in zip(allocation, group.itertuples(), strict=True):
                lots = int(
                    np.trunc(
                        weight
                        * s["initial_cash"]
                        * s["gross_notional_on_initial_cash"]
                        / row.reference_settle
                        / multipliers[row.code]
                    )
                )
                rows[book].append((day, row.front_contract, lots, pd.Timestamp(cutoff)))
    for book, expected in rows.items():
        actual = pd.read_csv(report / f"orders_{book}.csv", parse_dates=["date", "decision_at"])
        audit.same(f"{book} target count", len(actual), len(expected))
        for actual_row, expected_row in zip(
            actual.itertuples(index=False, name=None), expected, strict=True
        ):
            for field, a, b in zip(actual.columns, actual_row, expected_row, strict=True):
                audit.same(f"{book} target {field}", a, b)


def verify_book(directory, summary, bars, multipliers, initial_cash, annualization, audit):
    daily = pd.read_csv(directory / "daily.csv", parse_dates=["date"])
    trades = pd.read_csv(directory / "trades.csv", parse_dates=["date"])
    holdings = pd.read_csv(directory / "holdings.csv", parse_dates=["date"])
    rejected = pd.read_csv(directory / "rejected.csv", parse_dates=["date"])
    trade_days = {d: g for d, g in trades.groupby("date")}
    held_days = {d: g for d, g in holdings.groupby("date")}
    order_path = directory.parent.parent / f"orders_{directory.name}.csv"
    intended = pd.read_csv(order_path, parse_dates=["date"]) if order_path.exists() else None
    intended_days = {} if intended is None else {d: g for d, g in intended.groupby("date")}
    rejected_days = {d: g for d, g in rejected.groupby("date")}
    held, previous_marks = {}, {}
    cash = initial_cash
    expected_fees = 0.0
    equity_path = [cash]
    forced = False
    for record in daily.itertuples():
        date = record.date
        before = held.copy()
        overnight = sum(
            q * multipliers[bars[(date, c)].code] * (bars[(date, c)].open - previous_marks[c])
            for c, q in held.items()
        )
        opening_cash = cash + overnight
        fees = 0.0
        if date in trade_days:
            for t in trade_days[date].itertuples():
                raw = bars[(date, t.contract)]
                mult = multipliers[raw.code]
                old = held.get(t.contract, 0)
                delta = t.new_lots - old
                closed = min(abs(old), abs(delta)) if old * delta < 0 else 0
                fee = abs(delta) * raw.open * mult * summary.cost_bps / 10000
                for label, actual, expected in (
                    ("old lots", t.old_lots, old),
                    ("delta", t.delta_lots, delta),
                    ("closed", t.closed_lots, closed),
                    ("opened", t.opened_lots, abs(delta) - closed),
                    ("fill", t.fill_price, raw.open),
                    ("fee", t.fee, fee),
                    ("notional", t.notional, abs(delta) * raw.open * mult),
                ):
                    audit.same(f"{directory.name}/{date}/{t.contract}/{label}", actual, expected)
                if summary.fill_scenario == "block_flat_ohlc":
                    audit.same(
                        "flat-OHLC cannot trade",
                        len({raw.open, raw.high, raw.low, raw.close}) == 1,
                        False,
                    )
                fees += fee
                if t.new_lots:
                    held[t.contract] = t.new_lots
                else:
                    held.pop(t.contract, None)
        intraday = gross = net = opening_margin = 0.0
        for contract, lots in held.items():
            raw = bars[(date, contract)]
            mult = multipliers[raw.code]
            intraday += lots * mult * (raw.settle - raw.open)
            opening_margin += abs(lots) * raw.open * mult * summary.margin_rate
            signed = lots * raw.settle * mult
            gross += abs(signed)
            net += signed
            previous_marks[contract] = raw.settle
        after = opening_cash + intraday - fees
        margin = gross * summary.margin_rate
        actual_held = held_days.get(date, pd.DataFrame(columns=["contract", "lots"]))
        audit.same(
            f"{date} held identities", dict(zip(actual_held.contract, actual_held.lots)), held
        )
        if intended is not None:
            expected_book = before.copy()
            if forced or date == daily.date.iloc[-1]:
                desired = {c: 0 for c in before}
            else:
                planned = intended_days.get(date, pd.DataFrame(columns=["contract", "target_lots"]))
                desired = dict(zip(planned.contract, planned.target_lots))
            rejected_codes = set(rejected_days[date].code) if date in rejected_days else set()
            for code in {bars[(date, c)].code for c in desired}:
                if code in rejected_codes:
                    continue
                expected_book = {
                    c: q for c, q in expected_book.items() if bars[(date, c)].code != code
                }
                expected_book.update(
                    {c: int(q) for c, q in desired.items() if q and bars[(date, c)].code == code}
                )
            audit.same(f"{date} targets or atomic rejection", held, expected_book)
        for label, actual, expected in (
            ("previous cash", record.previous_equity, cash),
            ("opening cash", record.opening_equity, opening_cash),
            ("overnight", record.overnight_pnl, overnight),
            ("intraday", record.intraday_pnl, intraday),
            ("fees", record.fees, fees),
            ("equity", record.equity, after),
            ("opening margin", record.opening_reserved_margin, opening_margin),
            ("settlement margin", record.reserved_margin, margin),
            ("free cash", record.available_cash, after - margin),
            ("return", record.net_return, after / cash - 1),
            ("gross", record.gross_notional, gross),
            ("net", record.net_notional, net),
            ("margin call", record.margin_call, margin > after * 0.5),
        ):
            audit.same(f"{directory.name}/{date}/{label}", actual, expected)
        expected_fees += fees
        equity_path.append(after)
        cash = after
        forced = margin > after * 0.5
    audit.same("terminal flat", held, {})
    audit.same("summary final equity", summary.final_equity, cash)
    audit.same("summary fees", summary.fees, expected_fees)
    audit.same("summary legs", summary.trade_legs, len(trades))
    audit.same("summary rejections", summary.rejected_product_orders, len(rejected))
    audit.same("summary return", summary.total_return, cash / initial_cash - 1)
    audit.same(
        "summary annualization",
        summary.annualized_return,
        (cash / initial_cash) ** (annualization / len(daily)) - 1,
    )
    wealth = np.array(equity_path)
    audit.same(
        "summary drawdown", summary.max_drawdown, (wealth / np.maximum.accumulate(wealth) - 1).min()
    )
    # Independent terminal-flat identity avoids merely reusing daily PnL fields.
    cashflows = sum(t.delta_lots * t.fill_price * multipliers[t.code] for t in trades.itertuples())
    audit.same("terminal trade-cashflow identity", cash, initial_cash - cashflows - expected_fees)


def reconcile(report: Path, destination: Path) -> dict:
    run = json.loads((report / "run.json").read_text(encoding="utf-8"))
    audit = Audit()
    for name, entry in run["inputs"].items():
        audit.same(f"input hash {name}", sha(entry["path"]), entry["sha256"])
    for name, expected in run["output_sha256"].items():
        audit.same(f"output hash {name}", sha(report / name), expected)
    source = run["inputs"]
    bars_frame = pd.read_csv(source["bars"]["path"], parse_dates=["date"])
    curves = pd.read_csv(source["curves"]["path"], parse_dates=["date"])
    calendar = pd.read_csv(source["calendar"]["path"], parse_dates=["date"])
    verify_targets(report, run, curves, calendar, audit)
    actual_marks = pd.read_csv(
        report / "market_marks.csv",
        parse_dates=["date", "previous_signal_date", "expiry_signal_date"],
    )
    originals = {(r.date, r.contract): r for r in bars_frame.itertuples()}
    versions = {c: g.sort_values("date") for c, g in curves.groupby("front_contract")}
    for r in actual_marks.itertuples():
        raw = originals[(r.date, r.contract)]
        for field in ("open", "high", "low", "close", "settle"):
            audit.same(
                f"original {r.contract}/{r.date}/{field}", getattr(r, field), getattr(raw, field)
            )
        candidates = versions[r.contract]
        position = candidates.date.searchsorted(r.previous_signal_date, side="right") - 1
        if position < 0:
            raise ValueError("No dated expiry known before execution")
        prior = candidates.iloc[position]
        audit.same("expiry version date", r.expiry_signal_date, prior.date)
        audit.same("expiry version value", r.expiry, prior.front_expiry)
        audit.same("no future expiry version", r.expiry_signal_date <= r.previous_signal_date, True)
    summary = pd.read_csv(report / "summary.csv")
    s = run["protocol"]["study"]
    for row in summary.itertuples():
        verify_book(
            report / row.scenario / row.book,
            row,
            originals,
            run["protocol"]["multipliers"],
            s["initial_cash"],
            s["annualization_sessions"],
            audit,
        )
    comparisons = pd.read_csv(report / "comparisons.csv")
    for r in comparisons.itertuples():
        g = summary.loc[summary.scenario.eq(r.scenario)].set_index("book")
        audit.same(
            "paired net difference",
            r.blend_minus_momentum_return_on_initial_cash,
            g.loc["momentum_carry", "total_return"] - g.loc["momentum", "total_return"],
        )
    prior = pd.read_csv(source["prior_exploration"]["path"])
    registry = pd.read_csv(report / "registered_exploration.csv")
    pd.testing.assert_frame_equal(registry.iloc[:15][prior.columns], prior, check_dtype=False)
    audit.same("exploration family", len(registry), 16)
    destination.mkdir(parents=True, exist_ok=False)
    result = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "checks": audit.checks,
        "differences": len(audit.differences),
        "details": audit.differences[:40],
        "run_sha256": sha(report / "run.json"),
        "audit_code_sha256": sha(__file__),
        "report": str(report.resolve()),
        "actual_execution_certified": False,
        "independent_new_sample": False,
        "production_modules_imported": False,
    }
    (destination / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>市场账独立复核</title><h1>固定市场账独立复核</h1>'
        f"<p>{audit.checks:,}项检查，{len(audit.differences)}项差异。检查数不是独立统计样本数。</p>"
        "<p>从原行情和冻结曲线重建目标、费用、现金、敞口与保证金，不导入生产下单或记账模块。"
        '核对实现，未认证历史初发、实际成交或策略有效性。</p><a href="audit.json">完整记录</a><pre>'
        + escape("\n".join(audit.differences[:40]))
        + "</pre>",
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reconcile(args.report, args.output)
    print(json.dumps(result, ensure_ascii=False))
    if result["differences"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
