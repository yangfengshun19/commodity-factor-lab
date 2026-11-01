"""Build a small GitHub source bundle using an explicit file allowlist."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    "README.md",
    "CONTRIBUTING.md",
    "pyproject.toml",
    "MANIFEST.in",
    ".gitignore",
    ".gitattributes",
    ".env.example",
    "LICENSE",
    "LICENSE.md",
    "NOTICE",
    "configs/validation_plan.json",
}
PATTERNS = {
    "src/commodity_factor_lab": {".py"},
    "configs": {".toml"},
    "tests": {".py"},
    "scripts": {".py"},
    "docs": {".md", ".svg", ".png", ".jpg"},
    ".github": {".yml", ".yaml"},
    "research/legacy": {".py", ".md"},
}


def is_public_document(path: Path) -> bool:
    """Keep desktop exchanges and unfinished drafts out of release artifacts."""
    relative = path.relative_to(ROOT)
    if relative.parts[0] != "docs":
        return True
    if any(part in {"collaboration", "drafts"} for part in relative.parts[1:-1]):
        return False
    return "_draft" not in path.stem.lower() and not path.stem.lower().startswith("draft")


def public_files() -> list[Path]:
    files = [ROOT / name for name in ROOT_FILES if (ROOT / name).is_file()]
    files.extend([ROOT / "data/README.md", ROOT / "reports/README.md"])
    for folder, suffixes in PATTERNS.items():
        files.extend(
            path
            for path in (ROOT / folder).rglob("*")
            if path.is_file() and path.suffix in suffixes and is_public_document(path)
        )
    for path in files:
        if not path.resolve().is_relative_to(ROOT) or "__pycache__" in path.parts:
            raise ValueError(f"Unexpected source path: {path}")
    return sorted(set(files))


def main() -> None:
    output = ROOT / "dist/commodity-factor-lab-github.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    files = public_files()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in files:
            info = ZipInfo(
                "commodity-factor-lab/" + path.relative_to(ROOT).as_posix(),
                date_time=(2026, 1, 1, 0, 0, 0),
            )
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    print(f"Public source bundle: {output} ({len(files)} files, {output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
