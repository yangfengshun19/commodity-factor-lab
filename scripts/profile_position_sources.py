"""Reproduce the bounded SHFE broker-rank coverage profile from saved public responses."""

import argparse
from pathlib import Path

from commodity_factor_lab.positions import profile_position_sources

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--raw", type=Path, required=True)
parser.add_argument("--curves", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    print(profile_position_sources(args.raw, args.curves, args.output))
