"""Deterministic synthetic market used only to demonstrate the workflow."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import FAMILIES
from .factors import price_factors


def generate_demo(directory: Path, seed: int = 2025) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "SYNTHETIC_DATA.json"
    if any(directory.iterdir()) and not marker.exists():
        raise ValueError("Refusing to overwrite an unmarked data directory with synthetic data")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", "2024-12-31")
    products = [
        "a",
        "b",
        "c",
        "cs",
        "m",
        "y",
        "p",
        "rm",
        "oi",
        "sr",
        "cu",
        "al",
        "pb",
        "zn",
        "sn",
        "ni",
        "sc",
        "fu",
        "bu",
        "ru",
        "l",
        "v",
        "pp",
        "ta",
        "eg",
        "ma",
    ]
    common = rng.normal(0, 0.004, len(dates))
    for index, code in enumerate(products):
        innovations = rng.normal(0, 0.009, len(dates)) + common
        # Modest serial dependence is part of the synthetic process, not a claim
        # about any named real commodity or a fitted model of real market returns.
        for day in range(1, len(dates)):
            innovations[day] += 0.06 * innovations[day - 1]
        close = (700 + index * 270) * np.exp(innovations.cumsum())
        opening = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.002, len(dates)))
        high = np.maximum(opening, close) * np.exp(rng.uniform(0.001, 0.009, len(dates)))
        low = np.minimum(opening, close) * np.exp(-rng.uniform(0.001, 0.009, len(dates)))
        bars = price_factors(
            pd.DataFrame({"date": dates, "open": opening, "high": high, "low": low, "close": close})
        )
        for family in ("trend", "value", "volatility"):
            target = directory / family
            target.mkdir(exist_ok=True)
            bars[["date", "open", "high", "low", "close", *FAMILIES[family].factors]].to_csv(
                target / f"Synthetic_{code}_factors.csv",
                index=False,
                float_format="%.10g",
            )
    marker.write_text(
        json.dumps(
            {
                "synthetic": True,
                "seed": seed,
                "products": len(products),
                "sessions": len(dates),
                "purpose": "Software demonstration only; no market-performance claim.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
