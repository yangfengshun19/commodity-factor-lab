"""Verify that every migrated original is still present with accounted-for content."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    manifest = ROOT / "private/migration_manifest.csv"
    if not manifest.exists():
        print("No local migration manifest. A public clone contains no original private files.")
        return 0
    adjustments_path = ROOT / "private/migration_adjustments.json"
    adjustments = (
        json.loads(adjustments_path.read_text(encoding="utf-8"))
        if adjustments_path.exists()
        else {}
    )
    failures, count = [], 0
    with manifest.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            count += 1
            path = (ROOT / row["current_path"]).resolve()
            if not path.is_relative_to(ROOT):
                failures.append(f"Outside workspace: {row['current_path']}")
                continue
            if not path.is_file():
                failures.append(f"Missing: {row['current_path']}")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = adjustments.get(row["current_path"], {}).get(
                "current_sha256", row["original_sha256"]
            )
            if digest != expected:
                failures.append(f"Unexpected change: {row['current_path']}")
    for failure in failures:
        print(failure)
    print(
        f"Verified {count} originals; {len(adjustments)} documented sanitizations; {len(failures)} failures."
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
