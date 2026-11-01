"""Audit attribution with trade cashflows, without importing production attribution."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Checks:
    def __init__(self):
        self.count = 0
        self.differences = []

    def same(self, label, actual, expected, atol=1e-6):
        self.count += 1
        if isinstance(actual, (float, np.floating)) or isinstance(expected, (float, np.floating)):
            ok = bool(np.isclose(actual, expected, atol=atol, rtol=1e-11, equal_nan=True))
        else:
            ok = actual == expected
        if not ok:
            self.differences.append(f"{label}: {actual!r} != {expected!r}")


def cash_parts(bars, daily, trades, holdings, codes, multipliers):
    """End marked position - previous marked position - purchases - fees."""
    quotes = {(r.date, r.contract): r for r in bars.itertuples()}
    positions = {d: g for d, g in holdings.groupby("date")}
    fills = {d: g for d, g in trades.groupby("date")}
    previous, old_values, rows = {}, {}, []
    for day in daily.date:
        current_frame = positions.get(day, holdings.iloc[:0])
        current = dict(zip(current_frame.contract, current_frame.lots))
        today = fills.get(day, trades.iloc[:0])
        for code in codes:
            end_value = sum(
                q * quotes[(day, c)].settle * multipliers[code]
                for c, q in current.items()
                if quotes[(day, c)].code == code
            )
            beginning_value = sum(old_values[c] for c in previous if quotes[(day, c)].code == code)
            events = today.loc[today.code.eq(code)]
            purchase = sum(
                t.delta_lots * quotes[(day, t.contract)].open * multipliers[code]
                for t in events.itertuples()
            )
            fee = float(events.fee.sum())
            before = {c for c in previous if quotes[(day, c)].code == code}
            after = {c for c in current if quotes[(day, c)].code == code}
            reasons = set(events.reason)
            if not reasons:
                label = "no_trade"
            elif "margin_liquidation" in reasons:
                label = "margin_liquidation"
            elif "terminal_liquidation" in reasons:
                label = "terminal_liquidation"
            elif before and after and before != after:
                label = "roll_including_any_resize"
            elif not before and after:
                label = "entry"
            elif before and not after:
                label = "exit"
            else:
                label = "same_contract_resize_or_reverse"
            rows.append(
                {
                    "date": day,
                    "code": code,
                    "gross_pnl": end_value - beginning_value - purchase,
                    "fees": fee,
                    "net_pnl": end_value - beginning_value - purchase - fee,
                    "signed_notional": end_value,
                    "gross_notional": abs(end_value),
                    "event": label,
                    "trade_legs": len(events),
                    "traded_notional": float(events.notional.sum()),
                }
            )
        previous = current
        old_values = {
            c: q * quotes[(day, c)].settle * multipliers[quotes[(day, c)].code]
            for c, q in current.items()
        }
    return pd.DataFrame(rows)


def compare_frame(checks, actual, expected, keys, fields, label):
    a, e = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    checks.same(label + " keys", list(a.index), list(e.index))
    for key in e.index:
        for field in fields:
            checks.same(f"{label}/{key}/{field}", a.loc[key, field], e.loc[key, field])


def centered_ranks(values):
    values = np.asarray(values, dtype=float)
    ranks = np.array(
        [float((values < v).sum()) + (float((values == v).sum()) + 1) / 2 for v in values]
    )
    return ranks - ranks.mean()


def reconcile(report, destination):
    run = json.loads((report / "run.json").read_text(encoding="utf-8"))
    original = json.loads(Path(run["original_run"]).read_text(encoding="utf-8"))
    previous = Path(run["original_run"]).parent
    checks = Checks()
    for name, entry in run["inputs"].items():
        checks.same("input hash " + name, sha(entry["path"]), entry["sha256"])
    for name, expected in run["output_sha256"].items():
        checks.same("output hash " + name, sha(report / name), expected)
    bars = pd.read_csv(run["inputs"]["bars"]["path"], parse_dates=["date"])
    curve = pd.read_csv(run["inputs"]["curves"]["path"], parse_dates=["date"])
    actual = pd.read_csv(report / "daily_product_attribution.csv", parse_dates=["date"])
    cash = run["initial_cash"]
    scope = run["protocol"]["analysis"]
    study = original["protocol"]["study"]
    target = pd.read_csv(previous / "target_audit.csv", parse_dates=["date", "signal_date"])
    calendar = pd.read_csv(
        previous / "assumed_calendar.csv", parse_dates=["date", "previous_signal_date"]
    )
    expected_frames = []
    for book in scope["books"]:
        path = previous / scope["scenario"] / book
        daily, trades, holdings = (
            pd.read_csv(path / f"{n}.csv", parse_dates=["date"])
            for n in ("daily", "trades", "holdings")
        )
        parts = cash_parts(
            bars, daily, trades, holdings, study["codes"], original["protocol"]["multipliers"]
        )
        expected_frames.append(parts.assign(book=book))
        saved = actual.loc[actual.book.eq(book)]
        compare_frame(
            checks,
            saved,
            parts,
            ["date", "code"],
            [
                "gross_pnl",
                "fees",
                "net_pnl",
                "signed_notional",
                "gross_notional",
                "event",
                "trade_legs",
                "traded_notional",
            ],
            book + " independent cash",
        )
        gross = parts.groupby("date").gross_pnl.sum()
        net = parts.groupby("date").net_pnl.sum()
        wealth = cash + net.cumsum()
        for d, value in wealth.items():
            checks.same(
                "cash reconstructed from trade flows",
                value,
                daily.loc[daily.date.eq(d), "equity"].iloc[0],
            )
        matrix = parts.pivot(index="date", columns="code", values="net_pnl") / cash
        covariance = matrix.cov(ddof=1)
        contributions = covariance.sum(axis=1)
        variance = float(covariance.to_numpy().sum())
        risk = pd.read_csv(report / "risk_summary.csv").set_index("book").loc[book]
        checks.same(
            "net volatility",
            risk.cash_normalized_net_volatility,
            np.sqrt(variance * 252),
            atol=1e-12,
        )
        checks.same(
            "gross volatility",
            risk.cash_normalized_gross_volatility,
            (gross / cash).std(ddof=1) * np.sqrt(252),
            atol=1e-12,
        )
        checks.same("daily variance", risk.daily_cash_variance, variance, atol=1e-16)
        exposure = parts.groupby("date")[["gross_notional", "signed_notional"]].sum()
        checks.same(
            "average gross",
            risk.average_gross_exposure_on_initial_cash,
            exposure.gross_notional.mean() / cash,
            atol=1e-12,
        )
        checks.same(
            "average net",
            risk.average_net_exposure_on_initial_cash,
            exposure.signed_notional.mean() / cash,
            atol=1e-12,
        )
        variance_rows = pd.read_csv(report / "variance_contributions.csv")
        variance_rows = variance_rows.loc[variance_rows.book.eq(book)].set_index("code")
        for code in study["codes"]:
            checks.same(
                "variance contribution",
                variance_rows.loc[code, "variance_contribution"],
                contributions[code],
                atol=1e-16,
            )
            checks.same(
                "variance share",
                variance_rows.loc[code, "variance_share"],
                contributions[code] / variance,
                atol=1e-12,
            )
        values = [cash, *wealth]
        dates = [calendar.previous_signal_date.iloc[0], *daily.date]
        peak_value, peak_date, worst, peak_at_worst, trough_date = (
            cash,
            dates[0],
            0.0,
            dates[0],
            dates[0],
        )
        for date, value in zip(dates, values, strict=True):
            if value > peak_value:
                peak_value, peak_date = value, date
            dd = value / peak_value - 1
            if dd < worst:
                worst, peak_at_worst, trough_date = dd, peak_date, date
        peak_equity = values[dates.index(peak_at_worst)]
        window = pd.read_csv(report / "drawdown_windows.csv").set_index("book").loc[book]
        checks.same("drawdown peak", window.peak_date, str(peak_at_worst.date()))
        checks.same("drawdown trough", window.trough_date, str(trough_date.date()))
        checks.same("drawdown ratio", window.max_drawdown, worst, atol=1e-12)
        window_parts = (
            parts.loc[parts.date.gt(peak_at_worst) & parts.date.le(trough_date)]
            .groupby("code")
            .net_pnl.sum()
        )
        saved_window = pd.read_csv(report / "drawdown_contributions.csv")
        saved_window = saved_window.loc[saved_window.book.eq(book)].set_index("code")
        for code in study["codes"]:
            expected = window_parts.get(code, 0.0)
            checks.same(
                "drawdown cash component", saved_window.loc[code, "net_pnl_in_drawdown"], expected
            )
            checks.same(
                "drawdown ratio component",
                saved_window.loc[code, "drawdown_contribution"],
                expected / peak_equity,
                atol=1e-12,
            )
    expected = pd.concat(expected_frames, ignore_index=True)
    expected["year"] = expected.date.dt.year
    for name, keys in (
        ("product_attribution", ["book", "code"]),
        ("product_year_attribution", ["book", "year", "code"]),
        ("annual_attribution", ["book", "year"]),
    ):
        table = expected.groupby(keys)[["gross_pnl", "fees", "net_pnl"]].sum().reset_index()
        compare_frame(
            checks,
            pd.read_csv(report / f"{name}.csv"),
            table,
            keys,
            ["gross_pnl", "fees", "net_pnl"],
            name,
        )
    for name, keys in (
        ("paired_product_difference", ["code"]),
        ("paired_annual_difference", ["year"]),
    ):
        values = expected.groupby(["book", *keys])[["gross_pnl", "fees", "net_pnl"]].sum()
        diff = (values.loc["momentum_carry"] - values.loc["momentum"]).reset_index()
        diff["net_difference_on_initial_cash"] = diff.net_pnl / cash
        compare_frame(
            checks,
            pd.read_csv(report / f"{name}.csv"),
            diff,
            keys,
            ["gross_pnl", "fees", "net_pnl", "net_difference_on_initial_cash"],
            name,
        )
    fees = expected.groupby(["book", "event"])[["fees", "trade_legs"]].sum().reset_index()
    compare_frame(
        checks,
        pd.read_csv(report / "fees_by_event.csv"),
        fees,
        ["book", "event"],
        ["fees", "trade_legs"],
        "event fees",
    )
    structure = pd.read_csv(report / "target_structure.csv", parse_dates=["date"]).set_index(
        ["date", "book"]
    )
    directions = pd.read_csv(report / "target_directions.csv", parse_dates=["date"]).set_index(
        ["date", "code"]
    )
    for (date, book), g in target.groupby(["date", "book"]):
        w = g.weight.to_numpy()
        row = structure.loc[(date, book)]
        checks.same("target max weight", row.max_absolute_weight, max(abs(w)), atol=1e-12)
        checks.same("target squared weights", row.sum_squared_weights, float(w @ w), atol=1e-12)
        checks.same(
            "effective target names",
            row.effective_absolute_names,
            abs(w).sum() ** 2 / (w @ w),
            atol=1e-12,
        )
        f = curve.loc[curve.date.eq(g.signal_date.iloc[0])].set_index("code").loc[g.code]
        m, c = centered_ranks(f.momentum_126d), centered_ranks(f.carry)
        ratio = abs(m).sum() / abs((m + c) / 2).sum()
        checks.same("score normalization", row.score_normalization_ratio, ratio, atol=1e-12)
        if book == "momentum":
            blend = target.loc[target.date.eq(date) & target.book.eq("momentum_carry")].set_index(
                "code"
            )
            for t in g.itertuples():
                r = directions.loc[(date, t.code)]
                checks.same("unchanged momentum target", r.momentum_weight, t.weight, atol=1e-12)
                checks.same(
                    "unchanged blend target",
                    r.blend_weight,
                    blend.loc[t.code, "weight"],
                    atol=1e-12,
                )
    checks.same(
        "old registry exact bytes",
        (report / "registered_exploration.csv").read_bytes(),
        (previous / "registered_exploration.csv").read_bytes(),
    )
    checks.same("known family", run["known_exploration_family"], 16)
    destination.mkdir(parents=True, exist_ok=False)
    result = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks.count,
        "differences": len(checks.differences),
        "details": checks.differences[:40],
        "run_sha256": sha(report / "run.json"),
        "audit_code_sha256": sha(__file__),
        "production_attribution_imported": False,
        "new_prediction_test": False,
        "actual_execution_certified": False,
    }
    (destination / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><h1>盈亏归因独立核对</h1>'
        f"<p>{checks.count:,}项检查，{len(checks.differences)}差异。以逐笔交易现金流重建，未导入生产归因模块。</p>"
        '<p>核对算术，不认证实际成交、统计显著、因果机制或独立样本。</p><a href="audit.json">详细记录</a>',
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
