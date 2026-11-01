"""Audit two immutable price vintages without choosing or repairing prices."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from commodity_factor_lab.catalog import SECTOR_MAP, code_from_filename
from commodity_factor_lab.data import OHLC, normalize_snapshot, read_csv


def read_vintage(root: Path, start: str, end: str) -> tuple[dict, list]:
    frames, inputs = {}, []
    for path in sorted((root / "trend").glob("*.csv")):
        code = code_from_filename(path.name)
        if SECTOR_MAP[code] == "financial":
            continue
        if code in frames:
            raise ValueError(f"Duplicate product {code}")
        frame, _ = normalize_snapshot(read_csv(path), ())
        frames[code] = frame.loc[frame.date.between(start, end)].set_index("date")[OHLC]
        inputs.append(
            {"file": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    return frames, inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old", "new", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-10-09")
    args = parser.parse_args()
    old, old_inputs = read_vintage(args.old, args.start, args.end)
    new, new_inputs = read_vintage(args.new, args.start, args.end)
    records = []
    for code in sorted(set(old) | set(new)):
        a = old.get(code, pd.DataFrame(columns=OHLC))
        b = new.get(code, pd.DataFrame(columns=OHLC))
        shared = a.index.intersection(b.index)
        left, right = a.loc[shared].to_numpy(dtype=float), b.loc[shared].to_numpy(dtype=float)
        comparable = np.isfinite(left) & np.isfinite(right)
        changed = (comparable & ~np.isclose(left, right, rtol=1e-6, atol=1e-6)).any(axis=1)
        records.append(
            {
                "code": code,
                "old_rows": len(a),
                "new_rows": len(b),
                "shared_rows": len(shared),
                "changed_ohlc_rows": int(changed.sum()),
                "only_old_dates": ";".join(str(x.date()) for x in a.index.difference(b.index)),
                "only_new_dates": ";".join(str(x.date()) for x in b.index.difference(a.index)),
            }
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output.resolve() / stamp
    if any(destination.is_relative_to(p.resolve()) for p in (args.old, args.new)):
        raise ValueError("Audit output must be outside both inputs")
    destination.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(records).to_csv(destination / "revision.csv", index=False, encoding="utf-8-sig")
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "created_utc": stamp,
                "start": args.start,
                "end": args.end,
                "old_inputs": old_inputs,
                "new_inputs": new_inputs,
                "interpretation": "Disagreement does not establish which price is correct; no cross-source fills.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(destination)


if __name__ == "__main__":
    main()
