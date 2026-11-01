"""Independent saved-ledger checks; imports no production research modules."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def verify_book(folder, raw, dates, multipliers, initial, fraction, window, cost_bps, margin):
    daily = pd.read_csv(folder / "daily.csv", parse_dates=["date"])
    trades = pd.read_csv(folder / "trades.csv", parse_dates=["date"])
    holdings = pd.read_csv(folder / "holdings.csv", parse_dates=["date"])
    events = pd.read_csv(folder / "capacity_events.csv", parse_dates=["date"])
    marks = {(r.date, r.contract): r for r in raw.itertuples()}
    indices = {day: i for i, day in enumerate(dates)}
    errors, checks = [], 0

    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition and len(errors) < 100:
            errors.append(message)

    def close(a, b, label):
        check(math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-6), label)

    def cap(day, contract):
        i = indices[day]
        values = [
            getattr(marks.get((d, contract)), "volume", np.nan)
            for d in dates[max(0, i - window) : i]
        ]
        return (
            math.floor(sum(values) / window * fraction)
            if len(values) == window and np.isfinite(values).all()
            else 0
        )

    cash, quantities, prior_marks = float(initial), {}, {}
    grouped_trades = {d: g for d, g in trades.groupby("date")}
    grouped_holdings = {d: g for d, g in holdings.groupby("date")}
    grouped_events = {d: g for d, g in events.groupby("date")}
    for day in daily.itertuples():
        before = quantities.copy()
        # All original holdings earn settle-to-settle; changes earn fill-to-settle.
        pnl = sum(
            q
            * multipliers[marks[(day.date, c)].code]
            * (marks[(day.date, c)].settle - prior_marks[c])
            for c, q in before.items()
        )
        fees, deltas = 0.0, {}
        current_trades = grouped_trades.get(day.date, trades.iloc[:0])
        for trade in current_trades.itertuples():
            row = marks[(day.date, trade.contract)]
            delta = int(trade.delta_lots)
            close(trade.fill_price, row.open, f"{day.date}/{trade.contract}/fill")
            close(
                trade.fee,
                abs(delta) * abs(row.open) * multipliers[row.code] * cost_bps / 10000,
                f"{day.date}/{trade.contract}/fee",
            )
            check(
                trade.old_lots == quantities.get(trade.contract, 0),
                f"{day.date}/{trade.contract}/old_lots",
            )
            quantities[trade.contract] = quantities.get(trade.contract, 0) + delta
            check(
                trade.new_lots == quantities[trade.contract],
                f"{day.date}/{trade.contract}/new_lots",
            )
            deltas[trade.contract] = deltas.get(trade.contract, 0) + delta
            fees += trade.fee
            pnl += delta * multipliers[row.code] * (row.settle - row.open)
        for contract, delta in deltas.items():
            if fraction is not None:
                check(
                    abs(delta) <= cap(day.date, contract), f"{day.date}/{contract}/past_volume_cap"
                )
        quantities = {c: q for c, q in quantities.items() if q}
        saved = grouped_holdings.get(day.date, holdings.iloc[:0])
        check(
            dict(zip(saved.contract, saved.lots, strict=True)) == quantities, f"{day.date}/holdings"
        )
        check(not saved.code.duplicated().any(), f"{day.date}/unintended_multiple_contracts")
        cash += pnl - fees
        close(day.equity, cash, f"{day.date}/cash")
        close(day.fees, fees, f"{day.date}/daily_fee")
        expected_margin = sum(
            abs(q)
            * abs(marks[(day.date, c)].settle)
            * multipliers[marks[(day.date, c)].code]
            * margin
            for c, q in quantities.items()
        )
        close(day.reserved_margin, expected_margin, f"{day.date}/margin")
        for event in grouped_events.get(day.date, events.iloc[:0]).itertuples():
            check(
                event.filled_delta_lots == deltas.get(event.contract, 0),
                f"{day.date}/{event.contract}/event_fill",
            )
            check(
                event.remaining_lots
                == event.requested_target_lots - quantities.get(event.contract, 0),
                f"{day.date}/{event.contract}/remaining",
            )
            check(
                event.max_trade_lots == cap(day.date, event.contract),
                f"{day.date}/{event.contract}/budget",
            )
        prior_marks = {c: marks[(day.date, c)].settle for c in quantities}
    check(bool(daily.terminal_unliquidated.iloc[-1]) == bool(quantities), "terminal_flag")
    return {"checks": checks, "differences": errors, "terminal_positions": quantities}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    raw = pd.read_csv(root / run["protocol"]["paths"]["bars"], parse_dates=["date"])
    dates = list(
        pd.read_csv(root / run["protocol"]["paths"]["calendar"], parse_dates=["date"]).date
    )
    original = json.loads(
        (root / run["protocol"]["paths"]["source_run"]).read_text(encoding="utf-8")
    )
    study = run["protocol"]["study"]
    results = {}
    for row in pd.read_csv(args.run / "summary.csv").itertuples():
        key = f"{row.scenario}/{row.book}"
        if row.status != "completed":
            results[key] = {"status": "failed_scenario_recorded", "error": row.error}
            continue
        fraction = (
            None
            if row.scenario == "unlimited"
            else float(row.scenario.removeprefix("past_volume_"))
        )
        results[key] = verify_book(
            args.run / row.scenario / row.book,
            raw,
            dates,
            original["protocol"]["multipliers"],
            study["initial_cash"],
            fraction,
            study["volume_window_sessions"],
            study["cost_bps"],
            study["margin_rate"],
        )
    args.output.mkdir(parents=True, exist_ok=False)
    payload = {
        "books": results,
        "checks": sum(r.get("checks", 0) for r in results.values()),
        "differences": sum(len(r.get("differences", [])) for r in results.values()),
    }
    (args.output / "audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: payload[k] for k in ("checks", "differences")}))
    if payload["differences"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
