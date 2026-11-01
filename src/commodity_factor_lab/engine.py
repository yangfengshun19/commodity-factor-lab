"""Cross-sectional diagnostics and a next-open, unit-gross research portfolio.

Every return is indexed by its ending open. A close-t signal executes at open
t+1+publication_lag and earns open-to-open P&L ending at t+2+publication_lag.
No future labels are consulted when constructing weights.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ResearchConfig


class MissingHeldPrice(ValueError):
    """A held asset has no reliable next opening mark. No P&L is invented."""


@dataclass
class FactorResult:
    ic: pd.Series
    layers: pd.DataFrame
    ledger: pd.DataFrame
    weights: pd.DataFrame
    error: str = ""


def quantile_groups(values: pd.DataFrame, groups: int, min_assets: int) -> pd.DataFrame:
    """Keep ties together. Abstain when breadth or distinct values are insufficient."""
    count = values.notna().sum(axis=1)
    ranks = values.rank(axis=1, method="average")
    labels = np.floor((ranks - 0.5).div(count.replace(0, np.nan), axis=0) * groups)
    labels = labels.clip(lower=0, upper=groups - 1)
    eligible = (count >= min_assets) & (values.nunique(axis=1) >= groups)
    labels.loc[~eligible] = np.nan
    # Require both tails; all-equal or broad ties must not create a one-sided book.
    both = labels.eq(0).any(axis=1) & labels.eq(groups - 1).any(axis=1)
    labels.loc[~both] = np.nan
    return labels


def signal_weights(labels: pd.DataFrame, groups: int, direction: int = 1) -> pd.DataFrame:
    high, low = labels.eq(groups - 1), labels.eq(0)
    high_weights = high.div(high.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    low_weights = low.div(low.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return direction * 0.5 * (high_weights - low_weights)


def performance(returns: pd.Series, trading_days: int = 252) -> dict:
    returns = pd.Series(returns, dtype=float)
    if returns.empty or not np.isfinite(returns).all():
        return {
            name: np.nan
            for name in (
                "annual_return",
                "annual_volatility",
                "sharpe",
                "max_drawdown",
                "calmar",
                "total_return",
            )
        }
    if (returns <= -1).any():
        raise ValueError("Portfolio equity reached zero; compounding is invalid")
    wealth = (1 + returns).cumprod()
    annual_return = float(np.expm1(np.log1p(returns).sum() * trading_days / len(returns)))
    volatility = float(returns.std(ddof=1) * np.sqrt(trading_days))
    # Include initial equity 1.0, so a loss on the first day is a drawdown.
    peak = wealth.cummax().clip(lower=1.0)
    drawdown = float((wealth / peak - 1).min())
    return {
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe": float(returns.mean() / returns.std(ddof=1) * np.sqrt(trading_days))
        if volatility > 0
        else np.nan,
        "max_drawdown": drawdown,
        "calmar": annual_return / abs(drawdown) if drawdown < 0 else np.nan,
        "total_return": float(wealth.iloc[-1] - 1),
    }


def portfolio_ledger(
    weights: pd.DataFrame, asset_returns: pd.DataFrame, cost_rate: float
) -> pd.DataFrame:
    """Rebalance from drifted holdings; charge initial entries and subsequent exits.

    Weights are post-cost target exposures. The small fixed-point solve accounts for
    the fact that trading costs reduce the capital used to fund those targets.
    No terminal forced liquidation is assumed; the final open is a valuation mark.
    """
    asset_returns = asset_returns.reindex_like(weights)
    active_missing = weights.ne(0) & asset_returns.isna()
    if active_missing.any().any():
        row, col = np.argwhere(active_missing.to_numpy())[0]
        raise MissingHeldPrice(
            f"Missing opening price for held {weights.columns[col]} at {weights.index[row]:%Y-%m-%d}"
        )
    previous_drift = np.zeros(weights.shape[1])
    rows = []
    for target, returns in zip(weights.to_numpy(), asset_returns.fillna(0).to_numpy(), strict=True):
        gross = float(target @ returns)
        if gross <= -1:
            raise ValueError("Portfolio equity reached zero")
        turnover = float(np.abs(target - previous_drift).sum())
        for _ in range(50):
            updated = float(np.abs((1 - cost_rate * turnover) * target - previous_drift).sum())
            if abs(updated - turnover) < 1e-13:
                turnover = updated
                break
            turnover = updated
        cost = turnover * cost_rate
        net = (1 - cost) * (1 + gross) - 1
        rows.append((gross, net, cost, turnover, float(np.abs(target).sum()), float(target.sum())))
        previous_drift = target * (1 + returns) / (1 + gross)
    return pd.DataFrame(
        rows,
        index=weights.index,
        columns=[
            "gross_return",
            "net_return",
            "cost",
            "turnover",
            "gross_exposure",
            "net_exposure",
        ],
    )


def evaluate_factor(panel: pd.DataFrame, factor: str, config: ResearchConfig) -> FactorResult:
    opens = panel.pivot(index="date", columns="code", values="open").sort_index()
    values = panel.pivot(index="date", columns="code", values=factor).reindex_like(opens)
    # Reindexing to the full observed family calendar keeps missing sessions visible.
    steps = 2 + config.publication_lag_sessions
    labels = quantile_groups(values, config.groups, config.min_assets)
    plans = signal_weights(labels, config.groups, config.directions.get(factor, 1))
    # At execution open, an order without a valid opening price cannot be filled.
    entries = plans.shift(steps - 1).fillna(0).where(opens.notna(), 0)
    longs, shorts = entries.clip(lower=0), -entries.clip(upper=0)
    both = longs.sum(axis=1).gt(0) & shorts.sum(axis=1).gt(0)
    entries = 0.5 * longs.div(longs.sum(axis=1).replace(0, np.nan), axis=0).fillna(
        0
    ) - 0.5 * shorts.div(shorts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    entries.loc[~both] = 0
    held_weights = entries.shift(1).fillna(0)
    asset_returns = opens / opens.shift(1) - 1
    forward = opens.shift(-steps) / opens.shift(-(steps - 1)) - 1
    # Label dates before the split must not use test-period outcomes.
    horizon_end = pd.Series(opens.index, index=opens.index).shift(-steps)
    crosses_split = (opens.index < pd.Timestamp(config.test_start)) & (
        horizon_end >= pd.Timestamp(config.test_start)
    )
    forward.loc[crosses_split] = np.nan
    paired = values.notna() & forward.notna()
    x = values.where(paired).rank(axis=1, method="average")
    y = forward.where(paired).rank(axis=1, method="average")
    x = x.sub(x.mean(axis=1), axis=0)
    y = y.sub(y.mean(axis=1), axis=0)
    denominator = np.sqrt(x.pow(2).sum(axis=1) * y.pow(2).sum(axis=1)).replace(0, np.nan)
    ic = (
        ((x * y).sum(axis=1) / denominator)
        .where(paired.sum(axis=1) >= config.min_assets)
        .rename("rank_ic")
    )
    # These are gross signal-date diagnostics, not executable group portfolios.
    layers = pd.DataFrame(
        {
            f"Q{group + 1}": forward.where(labels.eq(group)).mean(axis=1)
            for group in range(config.groups)
        }
    )
    try:
        ledger = portfolio_ledger(held_weights, asset_returns, config.cost_rate)
        # Trim pre-investment cash; retain all later cash days for honest annualization.
        active = ledger["gross_exposure"].gt(0)
        if active.any():
            ledger = ledger.loc[active.idxmax() :].copy()
        else:
            ledger = ledger.iloc[:0].copy()
        ledger.index.name = "date"
        if not ledger.empty:
            timeline = pd.Series(opens.index, index=opens.index)
            ledger["signal_date"] = timeline.shift(steps).reindex(ledger.index)
            ledger["entry_date"] = timeline.shift(1).reindex(ledger.index)
        return FactorResult(ic, layers, ledger, held_weights)
    except MissingHeldPrice as error:
        return FactorResult(ic, layers, pd.DataFrame(), held_weights, str(error))


def summarize(result: FactorResult, config: ResearchConfig) -> list[dict]:
    records = []
    split = pd.Timestamp(config.test_start)
    for period in ("full", "train", "test"):
        ic = result.ic
        ledger = result.ledger
        if period != "full":
            ic = ic.loc[ic.index < split] if period == "train" else ic.loc[ic.index >= split]
            if not ledger.empty:
                # Assign a period by entry open; never attribute a pre-split entry
                # to the holdout just because its P&L ends on the split boundary.
                mask = (
                    ledger["entry_date"] < split
                    if period == "train"
                    else ledger["entry_date"] >= split
                )
                ledger = ledger.loc[mask]
        valid_ic = ic.dropna()
        ic_std = valid_ic.std(ddof=1)
        days = len(ledger)
        status = "ok"
        if result.error:
            status = "missing_held_price"
        elif ledger.empty:
            status = "no_executable_portfolio"
        elif days < config.min_history_days:
            status = "short_history"
        metrics = (
            performance(ledger["net_return"], config.trading_days)
            if days
            else performance(pd.Series(dtype=float))
        )
        gross_metrics = performance(ledger["gross_return"], config.trading_days) if days else {}
        records.append(
            {
                "period": period,
                "status": status,
                "days": days,
                "ic_days": len(valid_ic),
                "start": ledger.index.min().strftime("%Y-%m-%d") if days else "",
                "end": ledger.index.max().strftime("%Y-%m-%d") if days else "",
                "rank_ic": float(valid_ic.mean()),
                "ic_ir": float(valid_ic.mean() / ic_std) if ic_std > 0 else np.nan,
                "ic_positive_rate": float(valid_ic.gt(0).mean()) if len(valid_ic) else np.nan,
                "annual_turnover": float(ledger["turnover"].mean() * config.trading_days)
                if days
                else np.nan,
                "mean_gross_exposure": float(ledger["gross_exposure"].mean()) if days else np.nan,
                "error": result.error,
                "gross_annual_return": gross_metrics.get("annual_return", np.nan),
                "gross_sharpe": gross_metrics.get("sharpe", np.nan),
                "mean_cost_bps": float(ledger["cost"].mean() * 10000) if days else np.nan,
                **metrics,
            }
        )
    return records
