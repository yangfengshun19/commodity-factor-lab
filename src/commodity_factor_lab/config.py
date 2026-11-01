"""Validated TOML configuration; no machine-specific paths."""

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .catalog import FAMILIES


@dataclass(frozen=True)
class ResearchConfig:
    data: Path
    output: Path
    families: tuple[str, ...] = tuple(FAMILIES)
    include_financial: bool = False
    start: str = "2023-01-01"
    end: str = "2025-10-10"
    test_start: str = "2025-01-01"
    groups: int = 5
    min_assets: int = 5
    trading_days: int = 252
    publication_lag_sessions: int = 1
    fee_bps: float = 3.0
    slippage_bps: float = 2.0
    min_history_days: int = 60
    directions: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.families or set(self.families) - FAMILIES.keys():
            raise ValueError("families must contain known factor families")
        if len(set(self.families)) != len(self.families):
            raise ValueError("families must not contain duplicates")
        for name in (
            "groups",
            "min_assets",
            "trading_days",
            "publication_lag_sessions",
            "min_history_days",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.groups < 2 or self.min_assets < self.groups:
            raise ValueError("Require min_assets >= groups >= 2")
        if self.publication_lag_sessions < 0 or self.trading_days < 1 or self.min_history_days < 2:
            raise ValueError("Invalid lag, annualization, or minimum history")
        if any(not math.isfinite(x) or x < 0 for x in (self.fee_bps, self.slippage_bps)):
            raise ValueError("Costs must be finite and nonnegative")
        if self.cost_rate >= 0.1:
            raise ValueError("Combined one-way cost must be below 1000 bps")
        if pd.Timestamp(self.start) >= pd.Timestamp(self.end):
            raise ValueError("start must precede end")
        if not pd.Timestamp(self.start) < pd.Timestamp(self.test_start) <= pd.Timestamp(self.end):
            raise ValueError("test_start must fall inside the research period")
        known = {factor for family in FAMILIES.values() for factor in family.factors}
        if set(self.directions) - known or any(
            type(x) is not int or x not in (-1, 1) for x in self.directions.values()
        ):
            raise ValueError("directions must map known factors to +1 or -1")
        if self.data.resolve() == self.output.resolve() or self.output.resolve().is_relative_to(
            self.data.resolve()
        ):
            raise ValueError("Output must be outside the input data directory")

    @property
    def cost_rate(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 10000


def load_config(path: Path) -> ResearchConfig:
    path = path.resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    unknown = set(raw) - {"paths", "research", "directions"}
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    base = path.parent.parent
    paths = raw["paths"]
    if set(paths) != {"data", "output"}:
        raise ValueError("paths must contain exactly data and output")
    research = raw.get("research", {}).copy()
    if "families" in research:
        research["families"] = tuple(research["families"])
    return ResearchConfig(
        data=(base / paths["data"]).resolve(),
        output=(base / paths["output"]).resolve(),
        directions=raw.get("directions", {}),
        **research,
    )
