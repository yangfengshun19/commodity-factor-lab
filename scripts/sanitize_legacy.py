"""Remove embedded credentials and global TLS bypasses, recording local hashes."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest_path = ROOT / "private/migration_adjustments.json"
    changes = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    for path in (ROOT / "research/legacy").rglob("*.py"):
        original_bytes = path.read_bytes()
        original = path.read_text(encoding="utf-8-sig")
        has_database = bool(re.search(r"^DB_PASS\s*=", original, flags=re.MULTILINE))
        has_tls_bypass = bool(
            re.search(r"^ssl\._create_default_https_context\s*=", original, flags=re.MULTILINE)
        )
        if not has_database and not has_tls_bypass:
            continue
        updated = original
        if has_database and "import os\n" not in updated:
            updated = "import os\n" + updated
        defaults = {
            "DB_USER": "researcher",
            "DB_PASS": "",
            "DB_HOST": "127.0.0.1",
            "DB_PORT": "3306",
            "DB_NAME": "futures_data",
        }
        for key, default in defaults.items() if has_database else []:
            updated = re.sub(
                rf"^{key}\s*=.*$",
                f'{key} = os.environ.get("{key}", "{default}")',
                updated,
                flags=re.MULTILINE,
            )
        updated = re.sub(
            r"^ssl\._create_default_https_context\s*=.*\n", "", updated, flags=re.MULTILINE
        )
        updated = updated.replace("import ssl\n", "")
        if updated != original:
            relative = path.relative_to(ROOT).as_posix()
            path.write_text(updated, encoding="utf-8")
            previous = changes.get(relative, {})
            changes[relative] = {
                "current_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "previous_sha256": hashlib.sha256(original_bytes).hexdigest(),
                "previous_adjustment": previous,
                "reason": "Replace embedded DB settings with environment variables; restore TLS certificate verification.",
            }
    manifest_path.write_text(
        json.dumps(changes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Recorded legacy sanitizations: {len(changes)}")


if __name__ == "__main__":
    main()
