import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import package_public


class PublicPackageTests(unittest.TestCase):
    def test_release_keeps_instructions_but_excludes_local_records_and_drafts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowed = ["README.md", "docs/final/REPORT.md", "docs/final/REPRODUCE.md"]
            excluded = [
                "docs/collaboration/review.md",
                "docs/collaboration/screenshot.png",
                "docs/protocol_draft_v0_3.md",
                "docs/protocol_E_draft_v0.md",
                "docs/drafts/protocol.md",
                "reports/runs/local/summary.csv",
                "data/market/raw.json",
            ]
            for name in [*allowed, *excluded, "data/README.md", "reports/README.md"]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")
            with patch.object(package_public, "ROOT", root):
                actual = {p.relative_to(root).as_posix() for p in package_public.public_files()}
            self.assertTrue(set(allowed).issubset(actual))
            self.assertFalse(set(excluded) & actual)
