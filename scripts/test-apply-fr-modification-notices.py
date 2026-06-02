#!/usr/bin/env python3
"""Unit tests for apply-fr-modification-notices.py."""

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("apply-fr-modification-notices.py")
spec = importlib.util.spec_from_file_location("apply_fr_modification_notices", SCRIPT_PATH)
assert spec is not None
notices = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(notices)


class NoticeNormalizationTests(unittest.TestCase):
    def test_explicit_baseline_wins_over_base_ref(self) -> None:
        baseline = notices.derive_baseline(
            Path.cwd(),
            explicit_baseline="v9.8.7",
            base_ref="refs/heads/fr-v0.52.6.x",
        )

        self.assertEqual(baseline, "v9.8.7")

    def test_exact_notice_strip_preserves_adjacent_copyright(self) -> None:
        content = (
            "# NOTE:\n"
            "# This file has been modified by FortifyRoot.\n"
            "# Original source: https://github.com/traceloop/openllmetry\n"
            "\n"
            "# Copyright 2023 Traceloop\n"
            "import os\n"
        )

        normalized = notices.normalize_notice(content)

        self.assertEqual(normalized.count("This file has been modified by FortifyRoot."), 1)
        self.assertIn("# Copyright 2023 Traceloop\n", normalized)

    def test_notice_preserves_bom(self) -> None:
        normalized = notices.normalize_notice("\ufeffimport os\n")

        self.assertTrue(normalized.startswith("\ufeff# NOTE:\n"))

    def test_notice_preserves_crlf(self) -> None:
        normalized = notices.normalize_notice("import os\r\n")

        self.assertIn("# NOTE:\r\n", normalized)
        self.assertNotIn("# NOTE:\n", normalized.replace("\r\n", ""))

    def test_notice_stays_after_shebang_and_encoding(self) -> None:
        normalized = notices.normalize_notice(
            "#!/usr/bin/env python3\n"
            "# -*- coding: utf-8 -*-\n"
            "from __future__ import annotations\n"
        )

        self.assertTrue(
            normalized.startswith(
                "#!/usr/bin/env python3\n"
                "# -*- coding: utf-8 -*-\n"
                "# NOTE:\n"
            )
        )
        self.assertIn("from __future__ import annotations\n", normalized)

    def test_toml_notice_uses_toml_comment_syntax(self) -> None:
        normalized = notices.normalize_notice("[tool.poetry]\nname = \"demo\"\n")

        self.assertTrue(normalized.startswith("# NOTE:\n"))
        self.assertIn('[tool.poetry]\nname = "demo"\n', normalized)

    def test_baseline_derives_from_base_ref_in_detached_head_context(self) -> None:
        baseline = notices.derive_baseline(
            Path.cwd(),
            explicit_baseline=None,
            base_ref="refs/heads/fr-v0.52.6.x",
        )

        self.assertEqual(baseline, "v0.52.6")

    def test_modified_upstream_notice_files_include_toml_and_skip_fr_added_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = Path(tmp_dir)
            packages = repo / "packages" / "demo"
            packages.mkdir(parents=True)
            (packages / "existing.py").write_text("import os\n", encoding="utf-8")
            (packages / "pyproject.toml").write_text(
                "[project]\nname = \"demo\"\n",
                encoding="utf-8",
            )
            (packages / "README.md").write_text("demo\n", encoding="utf-8")

            self._git(repo, "init")
            self._git(repo, "config", "user.email", "test@example.com")
            self._git(repo, "config", "user.name", "Test User")
            self._git(repo, "add", ".")
            self._git(repo, "commit", "-m", "upstream baseline")
            self._git(repo, "tag", "v0.52.6")

            (packages / "existing.py").write_text("import sys\n", encoding="utf-8")
            (packages / "pyproject.toml").write_text(
                "[project]\nname = \"demo-fr\"\n",
                encoding="utf-8",
            )
            (packages / "fr_added.py").write_text("import json\n", encoding="utf-8")
            (packages / "README.md").write_text("demo fr\n", encoding="utf-8")
            self._git(repo, "add", ".")
            self._git(repo, "commit", "-m", "fr changes")

            paths = notices.modified_upstream_notice_files(repo, "v0.52.6")

            self.assertEqual(
                paths,
                [
                    Path("packages/demo/existing.py"),
                    Path("packages/demo/pyproject.toml"),
                ],
            )

    def _git(self, repo: Path, *args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        )


if __name__ == "__main__":
    unittest.main()
