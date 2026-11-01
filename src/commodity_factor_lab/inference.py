"""Dependence-aware diagnostics, not a substitute for independent new data.

HAC uses Bartlett weights and asymptotic normal p-values. Missing observations
retain their calendar slots with zero estimating scores; they are never compressed
into adjacent sessions. BY adjusts the complete declared hypothesis family.
"""

import math

import numpy as np
import pandas as pd


def hac_mean(values: pd.Series, lags: int = 10, min_obs: int = 60) -> dict:
    x = np.asarray(values, dtype=float)
    valid = np.isfinite(x)
    n = int(valid.sum())
    result = {
        "nobs": n,
        "mean": float(x[valid].mean()) if n else np.nan,
        "se": np.nan,
        "t": np.nan,
        "p": np.nan,
        "status": "insufficient_history",
    }
    if lags < 0 or min_obs < 2:
        raise ValueError("Require lags >= 0 and min_obs >= 2")
    if n < max(min_obs, lags + 2):
        return result
    scores = np.where(valid, x - result["mean"], 0.0)
    meat = float(scores @ scores)
    for k in range(1, min(lags + 1, len(scores))):
        meat += 2 * (1 - k / (lags + 1)) * float(scores[k:] @ scores[:-k])
    variance = max(meat, 0) / n**2 * n / (n - 1)
    if variance <= 1e-24:
        result["status"] = "degenerate_variance"
        return result
    se = math.sqrt(variance)
    stat = result["mean"] / se
    result.update(se=se, t=stat, p=math.erfc(abs(stat) / math.sqrt(2)), status="ok")
    return result


def by_adjust(pvalues: list[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Yekutieli q-values. Untestable planned hypotheses count as p=1."""
    original = np.asarray(pvalues, dtype=float)
    if np.any(np.isfinite(original) & ((original < 0) | (original > 1))):
        raise ValueError("p-values must be in [0, 1]")
    m = len(original)
    if not m:
        return np.array([], dtype=float)
    values = np.where(np.isfinite(original), original, 1.0)
    order = np.argsort(values, kind="stable")
    harmonic = np.sum(1 / np.arange(1, m + 1))
    adjusted = values[order] * m * harmonic / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    output = np.empty(m)
    output[order] = adjusted
    output[~np.isfinite(original)] = np.nan
    return output


def hac_alpha(
    returns: pd.Series,
    benchmarks: pd.DataFrame,
    lags: int = 10,
    min_obs: int = 252,
    trading_days: int = 252,
) -> dict:
    """HAC intercept against named benchmarks; no claim of a complete risk model."""
    frame = pd.concat([returns.rename("response"), benchmarks], axis=1)
    array = frame.to_numpy(dtype=float)
    valid = np.isfinite(array).all(axis=1)
    y = array[valid, 0]
    design = np.column_stack([np.ones(valid.sum()), array[valid, 1:]])
    n, k = design.shape
    result = {
        "nobs": n,
        "annual_intercept": np.nan,
        "t": np.nan,
        "p": np.nan,
        "status": "insufficient_history",
    }
    if n < max(min_obs, lags + k + 1):
        return result
    if np.linalg.matrix_rank(design) < k:
        result["status"] = "rank_deficient"
        return result
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    scores = np.zeros((len(frame), k))
    scores[valid] = design * residual[:, None]
    meat = scores.T @ scores
    for lag in range(1, min(lags + 1, len(frame))):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (lags + 1)) * (cross + cross.T)
    bread = np.linalg.inv(design.T @ design)
    covariance = bread @ meat @ bread * n / (n - k)
    if covariance[0, 0] <= 1e-24:
        result["status"] = "degenerate_variance"
        return result
    stat = float(beta[0] / np.sqrt(covariance[0, 0]))
    result.update(
        annual_intercept=float(beta[0] * trading_days),
        t=stat,
        p=math.erfc(abs(stat) / math.sqrt(2)),
        status="ok",
    )
    return result


def paired_block_interval(
    returns: pd.Series,
    benchmark: pd.Series,
    block: int = 20,
    repetitions: int = 1000,
    seed: int = 20261029,
    min_obs: int = 252,
) -> dict:
    """Paired moving-block percentile CI of mean daily return difference.

    Keeps cross-strategy dependence by using identical draw indices. Does not
    supply a null-test p-value or a simultaneous confidence band.
    """
    frame = pd.concat([returns, benchmark], axis=1)
    result = {"low": np.nan, "high": np.nan, "status": "insufficient_history"}
    if block < 1 or repetitions < 100:
        raise ValueError("Require block >= 1 and repetitions >= 100")
    if frame.isna().any().any():
        result["status"] = "incomplete_paired_calendar"
        return result
    delta = frame.iloc[:, 0].to_numpy() - frame.iloc[:, 1].to_numpy()
    if not np.isfinite(delta).all() or len(delta) < max(min_obs, block * 2):
        return result
    rng = np.random.default_rng(seed)
    starts = rng.integers(
        0, len(delta) - block + 1, size=(repetitions, math.ceil(len(delta) / block))
    )
    indices = (starts[..., None] + np.arange(block)).reshape(repetitions, -1)[:, : len(delta)]
    means = delta[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    result.update(low=float(low), high=float(high), status="ok")
    return result
