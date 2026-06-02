#!/usr/bin/env python3
"""Apply/check FortifyRoot modification notices in the OpenLLMetry fork.

The FR fork branches are based on an upstream Traceloop/OpenLLMetry tag.
For files that existed in that upstream tag and were modified by FR, Apache
2.0 requires a prominent notice that the file changed. This script keeps that
notice consistent for supported text source/metadata files under packages/.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


NOTICE_LINES = [
    "# NOTE:",
    "# This file has been modified by FortifyRoot.",
    "# Original source: https://github.com/traceloop/openllmetry",
    "",
]
NOTICE_SUFFIXES = {".py", ".toml"}

BRANCH_RE = re.compile(r"^fr-v(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)\.x$")
TAG_RE = re.compile(
    r"^fr-v(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)\.(?P<fr_patch>\d+)$"
)


def run_git(repo: Path, args: list[str], *, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed:\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


def derive_baseline(
    repo: Path,
    explicit_baseline: str | None,
    base_ref: str | None = None,
) -> str:
    if explicit_baseline:
        return explicit_baseline

    ref_match = _match_fr_branch_or_tag(base_ref or "")
    if ref_match:
        return _baseline_from_match(ref_match)

    branch = run_git(repo, ["branch", "--show-current"], check=False)
    branch_match = _match_fr_branch_or_tag(branch)
    if branch_match:
        return _baseline_from_match(branch_match)

    tag = run_git(repo, ["describe", "--tags", "--exact-match"], check=False)
    tag_match = _match_fr_branch_or_tag(tag)
    if tag_match:
        return _baseline_from_match(tag_match)

    raise RuntimeError(
        "Could not derive upstream baseline. Checkout a branch named "
        "fr-vA.B.C.x, checkout a tag named fr-vA.B.C.D, or pass "
        "--upstream-baseline vA.B.C."
    )


def _match_fr_branch_or_tag(ref: str) -> re.Match[str] | None:
    """Match FR fork branch/tag names, tolerating common git ref prefixes."""
    ref_name = ref.strip()
    for prefix in ("refs/heads/", "refs/tags/", "origin/"):
        if ref_name.startswith(prefix):
            ref_name = ref_name[len(prefix):]
            break
    return BRANCH_RE.match(ref_name) or TAG_RE.match(ref_name)


def _baseline_from_match(match: re.Match[str]) -> str:
    return (
        f"v{match.group('major')}.{match.group('minor')}.{match.group('patch')}"
    )


def verify_baseline(repo: Path, baseline: str) -> None:
    run_git(repo, ["rev-parse", "--verify", f"{baseline}^{{commit}}"])


def modified_upstream_notice_files(repo: Path, baseline: str) -> list[Path]:
    diff_output = run_git(
        repo,
        ["diff", "--name-only", "--diff-filter=AM", f"{baseline}..HEAD", "--", "packages"],
    )
    paths: list[Path] = []

    for line in diff_output.splitlines():
        path = Path(line)
        if path.suffix not in NOTICE_SUFFIXES:
            continue
        if not (repo / path).exists():
            continue
        existed_upstream = subprocess.run(
            ["git", "cat-file", "-e", f"{baseline}:{line}"],
            cwd=repo,
            text=True,
            capture_output=True,
        ).returncode == 0
        if existed_upstream:
            paths.append(path)

    return sorted(paths)


def normalize_notice(content: str) -> str:
    bom = "\ufeff" if content.startswith("\ufeff") else ""
    body_content = content[len(bom):] if bom else content
    newline = "\r\n" if "\r\n" in body_content else "\n"
    lines = body_content.splitlines(keepends=True)
    insert_at = _python_header_insert_index(lines)
    body = lines[:insert_at] + _strip_existing_modified_notice(lines[insert_at:])
    return (
        bom
        + "".join(body[:insert_at])
        + _notice_text(newline)
        + "".join(body[insert_at:])
    )


def _notice_text(newline: str) -> str:
    return newline.join(NOTICE_LINES) + newline


def _python_header_insert_index(lines: list[str]) -> int:
    index = 0
    if lines and lines[0].startswith("#!"):
        index = 1
    if index < len(lines) and re.match(r"^#.*coding[:=]\s*[-\w.]+", lines[index]):
        index += 1
    return index


def _strip_existing_modified_notice(lines: list[str]) -> list[str]:
    if len(lines) < len(NOTICE_LINES):
        return lines

    for offset, expected in enumerate(NOTICE_LINES):
        if lines[offset].strip() != expected:
            return lines
    return lines[len(NOTICE_LINES):]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply/check FR modification notices in fr-openllmetry-py."
    )
    parser.add_argument(
        "--upstream-baseline",
        help="Upstream tag to diff against, e.g. v0.52.6. If omitted, derive from current FR branch/tag.",
    )
    parser.add_argument(
        "--base-ref",
        help="FR fork base branch/tag ref to derive the upstream baseline from, e.g. fr-v0.52.6.x. Intended for pull-request CI.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check notices and exit nonzero if any file would change.",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]

    try:
        baseline = derive_baseline(repo, args.upstream_baseline, args.base_ref)
        verify_baseline(repo, baseline)
        files = modified_upstream_notice_files(repo, baseline)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    changed: list[Path] = []
    for path in files:
        file_path = repo / path
        content = file_path.read_text(encoding="utf-8")
        normalized = normalize_notice(content)
        if normalized == content:
            continue
        changed.append(path)
        if not args.check:
            file_path.write_text(normalized, encoding="utf-8")

    if changed:
        action = "would update" if args.check else "updated"
        print(f"FR modification notices {action} for {len(changed)} file(s):")
        for path in changed:
            print(f"  {path}")
        return 1 if args.check else 0

    print(
        f"FR modification notices are up to date for {len(files)} modified upstream file(s) "
        f"against {baseline}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
