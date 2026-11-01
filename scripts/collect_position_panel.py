"""Collect a frozen daily ranking schedule, preserving responses and failures."""

import argparse
from pathlib import Path

from commodity_factor_lab.position_ingest import fetch_positions

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=Path("configs/position_study.toml"))
parser.add_argument("--limit", type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    print(fetch_positions(args.config, args.limit))
