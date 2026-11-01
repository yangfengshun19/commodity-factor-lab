"""Read historical factor snapshots without modifying or filling the source."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import FAMILIES, SECTOR_MAP, code_from_filename
from .config import ResearchConfig

OHLC = ["open", "high", "low", "close"]


@dataclass
class LoadedFamily:
    panel: pd.DataFrame
    quality: list[dict]
    inputs: list[dict]


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gb18030")


def normalize_snapshot(frame: pd.DataFrame, factors: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    df = frame.copy()
    info = {"source_rows": len(df), "date_header_repaired": False}
    if "date" not in df and "Unnamed: 0" in df:
        # Repair only an unambiguous ISO date column, never a numeric row index.
        values = df["Unnamed: 0"].dropna().astype(str)
        if len(values) and values.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            df = df.rename(columns={"Unnamed: 0": "date"})
            info["date_header_repaired"] = True
    missing = {"date", *OHLC} - set(df)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    # Reject numeric epoch representations rather than silently interpreting ns.
    text_dates = df["date"].astype(str)
    valid_format = text_dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}(?: 00:00:00)?")
    df["date"] = pd.to_datetime(text_dates.where(valid_format), errors="coerce")
    bad_date = df["date"].isna() | (df["date"] < "2000-01-01")
    info["invalid_date_rows"] = int(bad_date.sum())
    df = df.loc[~bad_date].copy()
    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates; resolve the source instead of choosing an arbitrary row")
    missing_factors = [factor for factor in factors if factor not in df]
    info["missing_factor_columns"] = ",".join(missing_factors)
    for factor in missing_factors:
        df[factor] = np.nan
    for col in OHLC + list(factors):
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    bad_price = df[OHLC].isna().any(axis=1) | df[OHLC].le(0).any(axis=1)
    bad_price |= df["high"] < df[["open", "low", "close"]].max(axis=1)
    bad_price |= df["low"] > df[["open", "high", "close"]].min(axis=1)
    info["invalid_ohlc_rows"] = int(bad_price.sum())
    # Validate each opening mark independently: a later bad close/high/low must
    # not retroactively cancel a trade at a known valid opening price.
    # Keep dates so missing opening marks cannot become multi-session returns.
    for col in OHLC:
        df.loc[df[col].le(0), col] = np.nan
    df.loc[bad_price, list(factors)] = np.nan
    return df[["date", *OHLC, *factors]].sort_values("date"), info


def load_family(config: ResearchConfig, family: str) -> LoadedFamily:
    directory = config.data / family
    factors = FAMILIES[family].factors
    quality, inputs, frames = [], [], []
    paths = sorted(directory.glob("*.csv"))
    if not paths:
        quality.append(
            {"family": family, "file": "", "status": "missing_directory_or_files", "rows": 0}
        )
    for path in paths:
        relative = path.relative_to(config.data).as_posix()
        inputs.append(
            {
                "file": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
        record = {"family": family, "file": relative, "status": "ok", "rows": 0}
        try:
            code = code_from_filename(path.name)
            record["code"] = code
            df, info = normalize_snapshot(read_csv(path), factors)
            record.update(info)
            record["out_of_range_rows"] = int((~df["date"].between(config.start, config.end)).sum())
            df = df.loc[df["date"].between(config.start, config.end)].copy()
            df["code"] = code
            df["name"] = path.name.split("_")[0]
            df["sector"] = SECTOR_MAP[code]
            record["sector"] = SECTOR_MAP[code]
            record["rows"] = len(df)
            record["valid_open_rows"] = int(df["open"].notna().sum())
            record["start"] = df["date"].min().strftime("%Y-%m-%d") if len(df) else ""
            record["end"] = df["date"].max().strftime("%Y-%m-%d") if len(df) else ""
            for factor in factors:
                record[f"coverage_{factor}"] = float(df[factor].notna().mean()) if len(df) else 0.0
            if SECTOR_MAP[code] == "financial" and not config.include_financial:
                record["status"] = "excluded_financial"
            elif df.empty:
                record["status"] = "no_valid_dates_in_range"
            else:
                frames.append(df)
        except (ValueError, KeyError, pd.errors.ParserError) as error:
            record["status"] = "rejected"
            record["error"] = str(error)
        quality.append(record)
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not panel.empty:
        if panel.duplicated(["date", "code"]).any():
            raise ValueError(f"Multiple source files supply the same date/code in {family}")
        panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
        calendar = pd.DatetimeIndex(panel["date"].unique()).sort_values()
        gaps = {}
        for code, subset in panel.groupby("code"):
            expected = calendar[
                (calendar >= subset["date"].min()) & (calendar <= subset["date"].max())
            ]
            gaps[code] = len(expected.difference(subset["date"]))
        for record in quality:
            record["calendar_gap_rows"] = gaps.get(record.get("code"), 0)
    return LoadedFamily(panel=panel, quality=quality, inputs=inputs)
