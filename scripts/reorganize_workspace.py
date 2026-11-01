"""One-time, lossless migration of the original thesis workspace.

Only runs when the original folders exist. Records every original file's SHA256
and destination under private/. Does not initialize Git or contact any service.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "1价差类": "spread",
    "2仓单类": "warrant",
    "3期现类": "basis",
    "4量价趋势类": "trend",
    "5持仓类": "positioning",
    "6价值类": "value",
    "7波动类": "volatility",
}


def destination(path: Path) -> Path:
    parts = path.relative_to(ROOT).parts
    top = parts[0]
    if top.startswith("12-"):
        return Path("private/thesis").joinpath(*parts[1:])
    if top == "程序介绍.docx":
        return Path("private/notes/program_overview.docx")
    if top == "数据库测试":
        return Path("research/legacy/database").joinpath(*parts[1:])
    if top == "00分项分析":
        if path.suffix == ".py":
            return Path("research/legacy/sector_analysis").joinpath(*parts[1:])
        if (
            path.suffix == ".csv"
            and "analysis_summary" not in path.name
            and "performance_summary" not in path.name
        ):
            return Path("data/archive/sector_analysis").joinpath(*parts[1:])
        return Path("reports/archive/sector_analysis").joinpath(*parts[1:])
    if top in FAMILIES:
        family = FAMILIES[top]
        if path.suffix == ".py":
            return Path("research/legacy/analysis", family).joinpath(*parts[1:])
        if path.suffix == ".csv":
            if len(parts) == 2 and "panel_data" not in path.name:
                return Path("data/raw", family, path.name)
            return Path("data/archive", family).joinpath(*parts[1:])
        return Path("reports/archive", family).joinpath(*parts[1:])
    if len(parts) == 1 and path.suffix == ".py":
        return Path("research/legacy/collectors", path.name)
    raise ValueError(f"Unclassified original file: {path}")


def main() -> None:
    old_dirs = [ROOT / name for name in FAMILIES]
    old_dirs += [ROOT / "00分项分析", ROOT / "数据库测试"]
    old_dirs += list(ROOT.glob("12-*"))
    originals = list(ROOT.glob("*.py")) + list(ROOT.glob("*.docx"))
    for directory in old_dirs:
        if directory.is_dir():
            originals.extend(p for p in directory.rglob("*") if p.is_file())
    originals = sorted(set(originals))
    if not originals:
        print("No original folders remain; nothing to migrate.")
        return
    manifest = ROOT / "private/migration_manifest.csv"
    if manifest.exists():
        raise FileExistsError("A migration manifest already exists; refusing to overwrite it.")
    moves = []
    for source in originals:
        target = (ROOT / destination(source)).resolve()
        if not target.is_relative_to(ROOT) or target.exists():
            raise ValueError(f"Unsafe or existing destination: {target}")
        moves.append((source, target, hashlib.sha256(source.read_bytes()).hexdigest()))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["original_path", "current_path", "original_sha256", "bytes"])
        for source, target, digest in moves:
            target.parent.mkdir(parents=True, exist_ok=True)
            size = source.stat().st_size
            shutil.move(str(source), str(target))
            writer.writerow(
                [
                    source.relative_to(ROOT).as_posix(),
                    target.relative_to(ROOT).as_posix(),
                    digest,
                    size,
                ]
            )
    # Remove empty original directories only, never a nonempty tree.
    for directory in old_dirs:
        if directory.is_dir():
            for nested in sorted(directory.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if nested.is_dir() and not any(nested.iterdir()):
                    nested.rmdir()
            if not any(directory.iterdir()):
                directory.rmdir()
    print(f"Migrated {len(moves)} files. Manifest: private/migration_manifest.csv")


if __name__ == "__main__":
    main()
