"""Point-in-time fundamental revisions and within-product inventory features."""

import calendar as civil_calendar

import numpy as np
import pandas as pd

from .curve_factors import utc_column


def asof_fundamentals(
    observations: pd.DataFrame, calendar: pd.DataFrame, max_age_days: int = 10
) -> pd.DataFrame:
    required = {"code", "period_date", "available_at", "value", "unit", "source"}
    if required - set(observations) or {"date", "signal_cutoff"} - set(calendar):
        raise ValueError(
            "Fundamental values require economic dates, publication times, units and source"
        )
    data, sessions = observations.copy(), calendar.copy()
    data["available_at"] = utc_column(data["available_at"])
    data["period_date"] = pd.to_datetime(data["period_date"], errors="raise").dt.normalize()
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    sessions["date"] = pd.to_datetime(sessions["date"], errors="raise").dt.normalize()
    sessions["signal_cutoff"] = utc_column(sessions["signal_cutoff"])
    if max_age_days < 0 or not sessions.date.is_unique or not sessions.date.is_monotonic_increasing:
        raise ValueError("Invalid staleness policy or calendar")
    if data.duplicated(["code", "period_date", "available_at"]).any():
        raise ValueError("Duplicate fundamental publication key")
    if (
        data[["unit", "source", "code"]].isna().any().any()
        or data.groupby("code").unit.nunique().gt(1).any()
    ):
        raise ValueError("Mixed or missing units/source; convert explicitly before analysis")
    output = []
    for code, group in data.groupby("code"):
        for session in sessions.itertuples():
            known = group.loc[
                (group.available_at <= session.signal_cutoff) & (group.period_date <= session.date)
            ]
            if known.empty:
                continue
            current = known.sort_values(["period_date", "available_at"]).iloc[-1]
            age = (session.date - current.period_date).days
            output.append(
                {
                    "date": session.date,
                    "code": code,
                    "value": current.value if age <= max_age_days else np.nan,
                    "period_date": current.period_date,
                    "available_at": current.available_at,
                    "age_days": age,
                    "unit": current.unit,
                    "source": current.source,
                    "status": "ok" if age <= max_age_days else "stale",
                }
            )
    return pd.DataFrame(
        output,
        columns=[
            "date",
            "code",
            "value",
            "period_date",
            "available_at",
            "age_days",
            "unit",
            "source",
            "status",
        ],
    )


def seasonal_inventory(
    observations: pd.DataFrame,
    calendar: pd.DataFrame,
    lookback_years: int = 5,
    min_seasons: int = 3,
    min_observations: int = 60,
    seasonal_band_days: int = 30,
) -> pd.DataFrame:
    """Scarcity score: negative percentile against earlier-year seasonal observations.

    Revisions are selected separately at each historical information cutoff. Missing
    or short history is not repaired by ranks across unrelated commodity units.
    """
    if (
        lookback_years < min_seasons
        or min_seasons < 2
        or min_observations < 2
        or not 0 <= seasonal_band_days <= 90
    ):
        raise ValueError("Invalid seasonal history requirements")
    current = asof_fundamentals(observations, calendar)
    data, sessions = observations.copy(), calendar.copy()
    data["period_date"] = pd.to_datetime(data["period_date"]).dt.normalize()
    data["available_at"] = utc_column(data["available_at"])
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    if data.value.lt(0).any():
        raise ValueError("Inventory quantities cannot be negative")
    cutoffs = pd.Series(
        utc_column(sessions.signal_cutoff).to_numpy(), index=pd.to_datetime(sessions.date)
    )
    rows = []
    for row in current.itertuples():
        score = np.nan
        known = data.loc[(data.code == row.code) & (data.available_at <= cutoffs.loc[row.date])]
        known = known.sort_values("available_at").drop_duplicates("period_date", keep="last")
        # Earlier seasonal anniversaries avoid treating last December as an
        # independent historical season for a January observation.
        history = []
        for year in range(row.period_date.year - lookback_years, row.period_date.year):
            day = min(
                row.period_date.day, civil_calendar.monthrange(year, row.period_date.month)[1]
            )
            anchor = pd.Timestamp(year=year, month=row.period_date.month, day=day)
            seasonal = known.loc[
                (known.period_date - anchor).abs().dt.days <= seasonal_band_days
            ].copy()
            seasonal["season_year"] = year
            if not seasonal.empty:
                history.append(seasonal)
        known = (
            pd.concat(history, ignore_index=True)
            if history
            else known.iloc[:0].assign(season_year=pd.Series(dtype="int64"))
        )
        values = pd.to_numeric(known.value, errors="raise").dropna()
        if (
            row.status == "ok"
            and np.isfinite(row.value)
            and len(values) >= min_observations
            and known.loc[known.value.notna(), "season_year"].nunique() >= min_seasons
        ):
            percentile = ((values < row.value).sum() + 0.5 * (values == row.value).sum()) / len(
                values
            )
            score = float(0.5 - percentile)
        rows.append(
            {
                "date": row.date,
                "code": row.code,
                "inventory_scarcity": score,
                "seasonal_observations": len(values),
                "seasons": known.loc[known.value.notna(), "season_year"].nunique(),
                "publication_age_days": row.age_days,
                "status": "ok" if np.isfinite(score) else "insufficient_seasonal_history",
            }
        )
    return pd.DataFrame(rows)


def normalized_position_flow(
    net_position: pd.DataFrame, total_open_interest: pd.DataFrame, sessions: int = 20
) -> pd.DataFrame:
    if (
        sessions < 1
        or not net_position.index.equals(total_open_interest.index)
        or not net_position.columns.equals(total_open_interest.columns)
    ):
        raise ValueError("Position and OI must share the complete declared trading calendar")
    if not net_position.index.is_unique or not net_position.index.is_monotonic_increasing:
        raise ValueError("Position calendar must be unique and sorted")
    finite_positions = net_position.where(np.isfinite(net_position))
    valid = (
        finite_positions.notna()
        .rolling(sessions + 1, min_periods=sessions + 1)
        .sum()
        .eq(sessions + 1)
    )
    denominator = total_open_interest.where(
        total_open_interest.gt(0) & np.isfinite(total_open_interest)
    )
    return (finite_positions - finite_positions.shift(sessions)).div(denominator).where(valid)
