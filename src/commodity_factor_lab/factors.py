"""Pure, causal price factors. Input: one product's chronological OHLC bars.

Preserves the thesis factor names but requires complete rolling windows.
Other factor families are loaded as historical snapshots, not reverse-engineered
from OHLC. They require contract curves or point-in-time fundamental data.
"""

import numpy as np
import pandas as pd


def price_factors(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close"}
    if required - set(bars):
        raise ValueError(f"Missing columns: {sorted(required - set(bars))}")
    df = bars.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates in one product's bars")
    df = df.sort_values("date").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all() or (df[col] <= 0).any():
            raise ValueError("OHLC prices must be finite and positive")
    if (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
    ).any():
        raise ValueError("Inconsistent OHLC range")
    close = df["close"]
    ma20 = close.rolling(20, min_periods=20).mean()
    df["t_macross"] = close.rolling(10, min_periods=10).mean() / ma20
    df["t_mamom"] = close.rolling(60, min_periods=60).mean().diff(5)
    df["t_maratio"] = close / ma20
    df["t_mom"] = close / close.shift(20) - 1
    df["Val_halfyear"] = close / close.shift(126) - 1
    df["Val_oneyear"] = close / close.shift(252) - 1
    df["vol_std"] = np.log(close / close.shift(1)).rolling(20, min_periods=20).std()
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    df["vol_gk"] = np.sqrt((0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2).clip(lower=0))
    rs = np.log(df["high"] / df["close"]) * np.log(df["high"] / df["open"]) + np.log(
        df["low"] / df["close"]
    ) * np.log(df["low"] / df["open"])
    df["vol_rs"] = np.sqrt(rs.clip(lower=0))
    return df
