"""Past-volume execution budgets, not a model of opening-auction liquidity."""

import numpy as np
import pandas as pd

from .curve_factors import utc_column


def trailing_capacity(
    bars: pd.DataFrame, calendar: pd.DataFrame, fraction: float, window: int = 20
) -> pd.DataFrame:
    """Each date's budget uses exactly the preceding `window` calendar sessions.

    Zeros are genuine observations; missing sessions are never compressed or
    filled. An incomplete window permits zero lots. Current/future volume has
    no influence. Availability retains the original retrospective EOD assumption.
    """
    if type(window) is not int or window < 2 or not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Require a positive volume fraction and at least two sessions")
    data, sessions = bars.copy(), calendar.copy()
    for frame in (data, sessions):
        frame["date"] = pd.to_datetime(frame.date).dt.normalize()
    sessions["signal_cutoff"] = utc_column(sessions.signal_cutoff)
    if (
        not sessions.date.is_unique
        or not sessions.date.is_monotonic_increasing
        or not sessions.signal_cutoff.is_monotonic_increasing
        or data.duplicated(["date", "contract"]).any()
        or not data.date.isin(sessions.date).all()
    ):
        raise ValueError("Capacity needs a unique complete chronological calendar")
    data["volume"] = pd.to_numeric(data.volume, errors="raise")
    if data.volume.lt(0).any() or np.isinf(data.volume).any():
        raise ValueError("Volumes must be nonnegative finite observations or missing")
    volumes = data.pivot(index="date", columns="contract", values="volume").reindex(sessions.date)
    mean = volumes.rolling(window, min_periods=window).mean().shift(1)
    count = volumes.rolling(window, min_periods=1).count().shift(1).fillna(0)
    result = mean.stack(future_stack=True).rename("past_average_volume").reset_index()
    result["history_observations"] = count.stack(future_stack=True).to_numpy(dtype=int)
    result["max_trade_lots"] = np.floor(result.past_average_volume.fillna(0) * fraction).astype(int)
    lookup = sessions.set_index("date")
    result["capacity_available_at"] = result.date.map(lookup.signal_cutoff.shift(1))
    result["volume_window_end"] = result.date.map(
        pd.Series(sessions.date.shift(1).to_numpy(), index=sessions.date)
    )
    result["status"] = np.where(
        result.history_observations.eq(window), "ok", "incomplete_past_window"
    )
    return result
