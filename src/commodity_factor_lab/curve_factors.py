"""Causal contract-curve characteristics; no stitched-price momentum.

Positive carry means backwardation. Calendar contains explicit information
cutoffs. Retrospective mode permits revised EOD bars with unverified availability
and exposes that assumption; strict mode requires bar publication timestamps.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CurveRules:
    min_days: int = 30
    max_days: int = 540
    min_volume: float = 100
    min_open_interest: float = 1000
    momentum_sessions: int = 126
    basis_momentum_sessions: int = 252

    def __post_init__(self):
        if (
            not 0 < self.min_days < self.max_days
            or min(self.min_volume, self.min_open_interest) < 0
        ):
            raise ValueError("Invalid maturity/liquidity policy")
        if min(self.momentum_sessions, self.basis_momentum_sessions) < 2:
            raise ValueError("Full historical windows must contain at least two sessions")


def utc_column(values: pd.Series) -> pd.Series:
    if (
        values.isna().any()
        or not values.astype(str).str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).all()
    ):
        raise ValueError("Timestamps require explicit UTC offsets")
    return pd.to_datetime(values, utc=True, errors="raise")


def contract_curves(
    bars: pd.DataFrame,
    metadata: pd.DataFrame,
    calendar: pd.DataFrame,
    rules: CurveRules = CurveRules(),
    retrospective: bool = False,
) -> pd.DataFrame:
    required = {"date", "code", "contract", "settle", "volume", "open_interest"}
    if required - set(bars) or {"contract", "expiry", "metadata_available_at"} - set(metadata):
        raise ValueError("Contract bars and dated expiry metadata are required")
    if {"date", "signal_cutoff"} - set(calendar):
        raise ValueError("An explicit information-cutoff calendar is required")
    data, meta, sessions = bars.copy(), metadata.copy(), calendar.copy()
    # A pre-populated expiry may come from a later snapshot. Only the dated
    # metadata relation below is authoritative for historical eligibility.
    data = data.drop(columns=["expiry", "metadata_available_at", "signal_cutoff"], errors="ignore")
    for frame in (data, sessions):
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if not sessions.date.is_unique or not sessions.date.is_monotonic_increasing:
        raise ValueError("Session calendar must be unique and chronological")
    if data.duplicated(["date", "contract"]).any() or not data.date.isin(sessions.date).all():
        raise ValueError("Duplicate marks or bars outside the declared calendar")
    if data.groupby("contract").code.nunique().gt(1).any():
        raise ValueError("A contract cannot change its product identity")
    sessions["signal_cutoff"] = utc_column(sessions["signal_cutoff"])
    meta["metadata_available_at"] = utc_column(meta["metadata_available_at"])
    meta["expiry"] = pd.to_datetime(meta["expiry"], errors="raise").dt.normalize()
    if meta.duplicated(["contract", "metadata_available_at"]).any():
        raise ValueError("Duplicate publication keys in expiry metadata")
    for col in ("settle", "volume", "open_interest"):
        data[col] = pd.to_numeric(data[col], errors="raise")
    if data[["volume", "open_interest"]].lt(0).any().any():
        raise ValueError("Negative liquidity observations")
    if not retrospective and "available_at" not in data:
        raise ValueError("Strict curves require bar availability timestamps")
    if "available_at" in data:
        data["available_at"] = utc_column(data["available_at"])
    data = data.merge(sessions[["date", "signal_cutoff"]], on="date", validate="many_to_one")
    if "available_at" in data:
        data = data.loc[data["available_at"] <= data["signal_cutoff"]].copy()
    enriched = []
    for contract, group in data.groupby("contract"):
        publications = meta.loc[meta.contract.eq(contract)].sort_values("metadata_available_at")
        if publications.empty:
            continue
        joined = pd.merge_asof(
            group.sort_values("signal_cutoff"),
            publications[["expiry", "metadata_available_at"]],
            left_on="signal_cutoff",
            right_on="metadata_available_at",
            direction="backward",
        )
        if joined.expiry.isna().all():
            continue
        enriched.append(joined)
    eligible = (
        pd.concat(enriched, ignore_index=True)
        if enriched
        else data.assign(expiry=pd.NaT, metadata_available_at=pd.NaT)
    )
    eligible["days_to_expiry"] = (eligible["expiry"] - eligible["date"]).dt.days
    eligible = eligible.loc[
        eligible.days_to_expiry.between(rules.min_days, rules.max_days)
        & eligible.settle.gt(0)
        & np.isfinite(eligible.settle)
        & eligible.volume.ge(rules.min_volume)
        & eligible.open_interest.ge(rules.min_open_interest)
    ]
    selected = {
        (date, code): g.sort_values(["expiry", "contract"]).drop_duplicates("expiry").head(3)
        for (date, code), g in eligible.groupby(["date", "code"])
    }
    marks = bars.copy()
    marks["date"] = pd.to_datetime(marks["date"]).dt.normalize()
    marks = marks.set_index(["date", "contract"])
    records = []
    for code in sorted(bars.code.unique()):
        previous = None
        for session in sessions.itertuples():
            chosen = selected.get((session.date, code))
            row = {
                "date": session.date,
                "code": code,
                "signal_cutoff": session.signal_cutoff,
                "status": "insufficient_known_curve",
                "carry": np.nan,
                "relative_basis": np.nan,
                "rank1_return": np.nan,
                "rank2_return": np.nan,
                "bar_availability_checked": not retrospective,
            }
            # A return is always between two marks of yesterday's selected contract.
            # Today's changing rank or contract price level never creates a return.
            if previous is not None:
                for rank, old in enumerate(previous.itertuples(), 1):
                    if rank > 2:
                        break
                    key = (session.date, old.contract)
                    if key in marks.index:
                        current = marks.loc[key, "settle"]
                        timely = (retrospective and "available_at" not in marks) or (
                            "available_at" in marks
                            and pd.Timestamp(marks.loc[key, "available_at"])
                            <= session.signal_cutoff
                        )
                        if timely and np.isfinite(current) and current > 0 and old.settle > 0:
                            row[f"rank{rank}_return"] = float(current / old.settle - 1)
            if chosen is not None and len(chosen) >= 2:
                first, second = chosen.iloc[0], chosen.iloc[1]
                term = (second.expiry - first.expiry).days / 365.25
                row.update(
                    status="ok",
                    carry=float(np.log(first.settle / second.settle) / term),
                    front_contract=first.contract,
                    next_contract=second.contract,
                    front_expiry=first.expiry,
                    next_expiry=second.expiry,
                    reference_settle=float(first.settle),
                    metadata_available_at=chosen.metadata_available_at.max(),
                    liquidity_contracts=len(chosen),
                )
                if len(chosen) >= 3:
                    third = chosen.iloc[2]
                    farther = np.log(second.settle / third.settle) / (
                        (third.expiry - second.expiry).days / 365.25
                    )
                    row["relative_basis"] = float(row["carry"] - farther)
                    row["third_contract"] = third.contract
            records.append(row)
            previous = chosen
    result = pd.DataFrame(records).sort_values(["code", "date"])
    for col, window in (
        ("momentum_126d", rules.momentum_sessions),
        ("basis_momentum", rules.basis_momentum_sessions),
    ):
        first = result.groupby("code")["rank1_return"].transform(
            lambda s: (1 + s).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1
        )
        result[col] = first
        if col == "basis_momentum":
            second = result.groupby("code")["rank2_return"].transform(
                lambda s: (1 + s).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1
            )
            result[col] -= second
    result["risk_volatility"] = result.groupby("code")["rank1_return"].transform(
        lambda s: s.rolling(60, min_periods=60).std() * np.sqrt(252)
    )
    return result.sort_values(["date", "code"]).reset_index(drop=True)
