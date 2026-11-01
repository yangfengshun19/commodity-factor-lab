"""Fixed, intentionally small research protocol. No best-backtest model search."""

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SIGNALS = ("momentum_126d", "trend_10_60", "reversal_5d")


@dataclass(frozen=True)
class StudyConfig:
    data: Path
    output: Path
    start: str = "2023-01-01"
    end: str = "2025-10-09"
    training_sessions: int = 252
    test_sessions: int = 63
    history_sessions: int = 252
    publication_lag_sessions: int = 1
    rebalance_sessions: int = 5
    groups: int = 3
    min_assets: int = 6
    min_median_volume: float = 5000
    min_inference_obs: int = 252
    min_ic_obs: int = 60
    hac_lags: int = 10
    bootstrap_repetitions: int = 1000
    primary_cost_bps: float = 5.0
    cost_scenarios_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)
    sectors: tuple[str, ...] = (
        "all_commodities",
        "agriculture",
        "energy_chemicals",
        "nonferrous",
        "ferrous",
    )

    def __post_init__(self) -> None:
        integer_names = (
            "training_sessions",
            "test_sessions",
            "history_sessions",
            "publication_lag_sessions",
            "rebalance_sessions",
            "groups",
            "min_assets",
            "min_inference_obs",
            "min_ic_obs",
            "hac_lags",
            "bootstrap_repetitions",
        )
        if any(type(getattr(self, name)) is not int for name in integer_names):
            raise ValueError("Session counts and statistical settings must be integers")
        if self.training_sessions < 60 or self.history_sessions < 126 or self.test_sessions < 5:
            raise ValueError("Insufficient training, price history or fold length")
        if self.publication_lag_sessions < 0 or self.rebalance_sessions < 1:
            raise ValueError("Invalid publication lag or rebalancing interval")
        if self.groups < 2 or self.min_assets < 2 * self.groups:
            raise ValueError("Require at least two names per group in the minimum universe")
        if self.hac_lags < self.rebalance_sessions or self.min_ic_obs < self.hac_lags + 2:
            raise ValueError(
                "HAC horizon must cover holding-label overlap with enough observations"
            )
        if self.min_inference_obs < 60 or self.bootstrap_repetitions < 100:
            raise ValueError("Invalid inference sample or bootstrap repetitions")
        if not self.cost_scenarios_bps or len(set(self.cost_scenarios_bps)) != len(
            self.cost_scenarios_bps
        ):
            raise ValueError("Cost scenarios must be nonempty and unique")
        if self.primary_cost_bps not in self.cost_scenarios_bps:
            raise ValueError("Primary cost must be one of the declared scenarios")
        if any(not math.isfinite(c) or not 0 <= c < 1000 for c in self.cost_scenarios_bps):
            raise ValueError("Invalid cost scenario")
        if not math.isfinite(self.min_median_volume) or self.min_median_volume < 0:
            raise ValueError("Invalid lagged volume threshold")
        known = {
            "all_commodities",
            "agriculture",
            "energy_chemicals",
            "nonferrous",
            "ferrous",
            "precious",
            "new_materials",
            "other",
        }
        if (
            not self.sectors
            or set(self.sectors) - known
            or len(set(self.sectors)) != len(self.sectors)
        ):
            raise ValueError("Unknown or duplicate sector")
        if pd.isna(pd.Timestamp(self.start)) or not pd.Timestamp(self.start) < pd.Timestamp(
            self.end
        ):
            raise ValueError("Invalid study dates")
        source, destination = self.data.resolve(), self.output.resolve()
        if source == destination or destination.is_relative_to(source):
            raise ValueError("Study output must be outside its input directory")


def load_protocol(path: Path) -> StudyConfig:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    if set(raw) != {"paths", "study"} or set(raw["paths"]) != {"data", "output"}:
        raise ValueError("Expected exactly paths/data/output and study configuration")
    base = path.resolve().parent.parent
    settings = raw["study"]
    for field in ("cost_scenarios_bps", "sectors"):
        if field in settings:
            settings[field] = tuple(settings[field])
    return StudyConfig(
        data=(base / raw["paths"]["data"]).resolve(),
        output=(base / raw["paths"]["output"]).resolve(),
        **settings,
    )
