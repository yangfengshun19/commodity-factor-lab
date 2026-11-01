"""Build same-contract opening returns with a timestamped, lagged roll rule.

Requires an explicit trading-session calendar, real expiries and data availability
timestamps. No month-15 approximation, continuous-price stitching or price filling.
The output is an input to research, not a margin/settlement execution simulator.
"""

from datetime import timedelta

import numpy as np
import pandas as pd


def same_contract_panel(
    bars: pd.DataFrame, calendar: pd.DataFrame, roll_days: int = 20
) -> pd.DataFrame:
    required = {
        "date",
        "code",
        "contract",
        "open",
        "volume",
        "open_interest",
        "multiplier",
        "expiry",
        "available_at",
    }
    if required - set(bars) or {"date", "open_at", "signal_cutoff"} - set(calendar):
        raise ValueError(
            "Contract bars require explicit identity/expiry/multiplier/available_at; calendar requires date/open_at/signal_cutoff"
        )
    if type(roll_days) is not int or roll_days < 1:
        raise ValueError("roll_days must be a positive integer")
    data, sessions = bars.copy(), calendar.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise").dt.normalize()
    data["expiry"] = pd.to_datetime(data["expiry"], errors="raise").dt.normalize()
    sessions["date"] = pd.to_datetime(sessions["date"], errors="raise").dt.normalize()
    # UTC or explicit offsets are mandatory; naive timestamps do not establish availability.
    for frame, col in ((data, "available_at"), (sessions, "open_at"), (sessions, "signal_cutoff")):
        if (
            frame[col].isna().any()
            or not frame[col].astype(str).str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).all()
        ):
            raise ValueError(f"{col} must carry an explicit timezone offset")
        frame[col] = pd.to_datetime(frame[col], utc=True, errors="raise")
    if sessions["date"].isna().any() or data[["date", "expiry"]].isna().any().any():
        raise ValueError("Missing session date or expiry")
    if not sessions["date"].is_monotonic_increasing or sessions["date"].duplicated().any():
        raise ValueError("Session calendar must be sorted and unique")
    if not sessions["open_at"].is_monotonic_increasing or sessions["open_at"].duplicated().any():
        raise ValueError("Actual opening timestamps must be strictly increasing")
    if (sessions["signal_cutoff"] <= sessions["open_at"]).any():
        raise ValueError("Signal cutoff must follow that trading session's opening")
    if (
        sessions["signal_cutoff"].iloc[:-1].to_numpy() >= sessions["open_at"].iloc[1:].to_numpy()
    ).any():
        raise ValueError("A session's signal cutoff must precede the next opening")
    for col in ("code", "contract"):
        if data[col].isna().any() or data[col].astype(str).str.strip().eq("").any():
            raise ValueError("Missing contract identity")
    data["code"] = data["code"].astype(str).str.lower()
    if data.duplicated(["date", "code", "contract"]).any():
        raise ValueError("Duplicate contract marks")
    if not data["date"].isin(sessions["date"]).all():
        raise ValueError("Contract data contains dates outside the declared calendar")
    for col in ("open", "volume", "open_interest", "multiplier"):
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not np.isfinite(data[col]).all():
            raise ValueError(f"Nonfinite {col}")
    if data["multiplier"].le(0).any() or data[["volume", "open_interest"]].lt(0).any().any():
        raise ValueError("Invalid contract multiplier, volume or open interest")
    if (data.groupby(["code", "contract"])["expiry"].nunique() > 1).any():
        raise ValueError("Conflicting true expiry dates")
    marks = data.set_index(["date", "code", "contract"])
    previous = {}
    records = []
    for position in range(1, len(sessions) - 1):
        signal, entry, exit_session = (
            sessions.iloc[position - 1],
            sessions.iloc[position],
            sessions.iloc[position + 1],
        )
        known = data.loc[
            (data["date"] == signal["date"])
            & (data["available_at"] <= signal["signal_cutoff"])
            & (data["volume"] > 0)
            & (data["open_interest"] > 0)
            & (data["expiry"] > entry["date"].to_pydatetime() + timedelta(days=roll_days))
        ]
        for code in sorted(data["code"].unique()):
            pool = known.loc[known["code"] == code].sort_values(
                ["open_interest", "expiry", "contract"], ascending=[False, True, True]
            )
            row = {
                "date": exit_session["date"],
                "signal_date": signal["date"],
                "entry_date": entry["date"],
                "code": code,
                "status": "no_known_eligible_contract",
                "return": np.nan,
                "one_contract_pnl": np.nan,
            }
            if pool.empty:
                records.append(row)
                continue
            selected = pool.iloc[0]
            contract = selected["contract"]
            row.update(
                contract=contract,
                selection_available_at=selected["available_at"],
                rolled=code in previous and previous[code] != contract,
                entry_trade_legs=1
                if code not in previous
                else (2 if previous[code] != contract else 0),
            )
            first, last = (entry["date"], code, contract), (exit_session["date"], code, contract)
            if first not in marks.index or last not in marks.index:
                row["status"] = "missing_same_contract_mark"
            elif marks.loc[first, "multiplier"] != marks.loc[last, "multiplier"]:
                row["status"] = "multiplier_changed"
            else:
                opening, closing = marks.loc[first, "open"], marks.loc[last, "open"]
                row.update(
                    entry_open=opening,
                    exit_open=closing,
                    multiplier=marks.loc[first, "multiplier"],
                    one_contract_pnl=float((closing - opening) * marks.loc[first, "multiplier"]),
                    status="ok" if opening > 0 else "cash_pnl_only_nonpositive_price",
                )
                if opening > 0:
                    row["return"] = float(closing / opening - 1)
                previous[code] = contract
            records.append(row)
    return pd.DataFrame(records)
