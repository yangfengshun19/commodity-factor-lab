"""Scheduled nominal-exposure proxy portfolio with explicit capital accounting.

This engine requires valid same-asset returns. Continuous-series snapshots do
not become tradable futures contract returns by passing through this engine.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import MissingHeldPrice


@dataclass
class Simulation:
    daily: pd.DataFrame
    weights: pd.DataFrame


def simulate(
    targets: pd.DataFrame,
    returns: pd.DataFrame,
    rebalance: pd.Series,
    cost_rate: float,
    liquidate: bool = True,
) -> Simulation:
    if not 0 <= cost_rate < 0.1:
        raise ValueError("Invalid one-way cost rate")
    if not targets.index.is_monotonic_increasing or not targets.index.is_unique:
        raise ValueError("Portfolio calendar must be sorted and unique")
    if not np.isfinite(targets.to_numpy()).all():
        raise ValueError("Target weights must be finite")
    if targets.abs().sum(axis=1).gt(1 + 1e-9).any():
        raise ValueError("Target gross exposure exceeds the unit-capital protocol")
    returns = returns.reindex_like(targets)
    schedule = rebalance.reindex(targets.index)
    if schedule.isna().any():
        raise ValueError("Missing rebalance schedule")
    drift = np.zeros(targets.shape[1])
    output, actual = [], []
    for number, (date, plan) in enumerate(targets.iterrows()):
        target = plan.to_numpy().copy() if bool(schedule.loc[date]) else drift.copy()
        marks = returns.loc[date].to_numpy()
        missing = (target != 0) & ~np.isfinite(marks)
        if missing.any():
            code = targets.columns[np.flatnonzero(missing)[0]]
            raise MissingHeldPrice(f"Missing held return for {code} at {date}")
        gross = float(target @ np.nan_to_num(marks, nan=0))
        if gross <= -1:
            raise ValueError("Portfolio equity exhausted")
        turnover = float(np.abs(target - drift).sum()) if schedule.loc[date] else 0.0
        if schedule.loc[date]:
            for _ in range(100):
                updated = float(np.abs((1 - cost_rate * turnover) * target - drift).sum())
                if abs(updated - turnover) < 1e-12:
                    turnover = updated
                    break
                turnover = updated
            else:
                raise ValueError("Trading-cost solve did not converge")
        entry_cost = cost_rate * turnover
        drift = target * (1 + np.nan_to_num(marks, nan=0)) / (1 + gross)
        exit_turnover = (
            float(np.abs(drift).sum()) if liquidate and number == len(targets) - 1 else 0.0
        )
        exit_cost = exit_turnover * cost_rate
        if entry_cost >= 1 or exit_cost >= 1:
            raise ValueError("Trading costs exhaust portfolio equity")
        net = (1 - entry_cost) * (1 + gross) * (1 - exit_cost) - 1
        output.append(
            (
                gross,
                net,
                entry_cost,
                exit_cost,
                turnover + exit_turnover,
                np.abs(target).sum(),
                target.sum(),
                bool(schedule.loc[date]),
            )
        )
        actual.append(target)
    daily = pd.DataFrame(
        output,
        index=targets.index,
        columns=[
            "gross_return",
            "net_return",
            "entry_cost",
            "terminal_cost",
            "turnover",
            "gross_exposure",
            "net_exposure",
            "rebalanced",
        ],
    )
    daily.index.name = "date"
    weights = pd.DataFrame(actual, index=targets.index, columns=targets.columns)
    return Simulation(daily, weights)
