#!/usr/bin/env bash

# Examples:
# Run every discovered package test suite:
#   ./scripts/run-tests.sh
#   ./scripts/run-tests.sh --all                            # explicit alias
# Run only tests marked with `@pytest.mark.fr` across matching packages:
#   ./scripts/run-tests.sh --fr
# Run only VCR cassette tests (replay mode, no API keys needed):
#   ./scripts/run-tests.sh --cassettes
# Run cassettes for a single package:
#   ./scripts/run-tests.sh --cassettes --package '*openai*'
# Run cassettes in recording mode (requires API keys):
#   ./scripts/run-tests.sh --cassettes -- --record-mode=all
# Run all tests for packages whose basename matches a glob:
#   ./scripts/run-tests.sh --package '*openai*'
# Run only tests marked with `@pytest.mark.fr` for a selected package glob:
#   ./scripts/run-tests.sh --fr --package '*anthropic*'
# List all discovered packages without running tests:
#   ./scripts/run-tests.sh --list
# Override the per-package test timeout:
#   PACKAGE_TEST_TIMEOUT_SECONDS=1800 ./scripts/run-tests.sh --fr
# Forward extra pytest arguments after `--`:
#   ./scripts/run-tests.sh --fr -- -x

set -euo pipefail

# Deactivate any inherited virtualenv.  An active VIRTUAL_ENV from a parent
# shell (e.g. fortifyroot-sdk-py/.venv) causes uv to resolve against the
# wrong environment, silently skipping test group and instruments deps.
unset VIRTUAL_ENV

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGES_DIR="$ROOT_DIR/packages"
REPORTS_ROOT="$ROOT_DIR/reports/test-run"
TIMESTAMP="$(date +"%Y%m%d-%H%M%S")"
REPORT_DIR="$REPORTS_ROOT/$TIMESTAMP"
STATE_DIR="$REPORT_DIR/state"
PACKAGE_TEST_TIMEOUT_SECONDS="${PACKAGE_TEST_TIMEOUT_SECONDS:-1200}"

MODE="all"
PACKAGE_FILTER=""
PYTHON_VERSION=""
LIST_ONLY=0
PYTEST_ARGS=()

PACKAGE_NAMES=()
PLATFORM="$(uname -s)"   # Darwin | Linux

skip_reason() {
  # Returns a non-empty reason string if the package should be skipped on
  # the current platform + Python version combination, or empty if OK.
  local pkg="$1"
  # Use explicit --python version if set, otherwise detect from default python3.
  local pyver="${PYTHON_VERSION:-$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)}"

  # --- Platform-only skips (macOS x86_64 missing wheels) ---
  if [[ "$PLATFORM" == "Darwin" ]]; then
    case "$pkg" in
      # ChromaDB default embeddings use ONNX + CoreML which crashes on macOS.
      opentelemetry-instrumentation-chromadb) echo "ONNX CoreML crashes on macOS"; return ;;
      # CrewAI → crewai-tools → lancedb: no macOS x86_64 wheel.
      opentelemetry-instrumentation-crewai) echo "crewai-tools dep lancedb has no macOS x86_64 wheel"; return ;;
    esac
  fi

  # --- Python-version-specific skips (upstream deps incompatible) ---
  case "$pkg" in
    # watsonx: ibm-watson-machine-learning → pandas 1.5.3 (no Python 3.12+ support)
    opentelemetry-instrumentation-watsonx)
      if [[ "$pyver" == 3.12* || "$pyver" == 3.13* ]]; then
        echo "ibm-watson-machine-learning pins pandas<2 (no Python $pyver support)"; return
      fi ;;
    # writer: writer SDK → watchdog 3.0 (C build fails on Python 3.12)
    opentelemetry-instrumentation-writer)
      if [[ "$pyver" == 3.12* ]]; then
        echo "writer SDK pins watchdog 3.0 (C build fails on Python 3.12)"; return
      fi ;;
    # milvus: milvus_lite imports pkg_resources which isn't in all environments.
    # Fails on macOS and in containers without setuptools.
    opentelemetry-instrumentation-milvus)
      echo "milvus_lite requires pkg_resources (upstream dep issue)"; return ;;
  esac

  echo ""
}

usage() {
  cat <<'EOF'
Usage: scripts/run-tests.sh [options] [-- <extra pytest args>]

Options:
  --fr               Run only tests marked with @pytest.mark.fr
  --cassettes        Run only VCR cassette tests (@pytest.mark.vcr).
                     Implies --record-mode=none unless overridden via --.
  --all              Run all tests (UT + FR + VCR). Same as default (no flags).
  --package <glob>   Restrict packages by basename glob, e.g. "*openai*"
  --python <ver>     Use a specific Python version, e.g. "3.12". Passed to uv.
  --list             List discovered packages and exit
  -h, --help         Show this help

Each package runs in its own isolated venv via `uv sync` + `uv run pytest`.
This avoids dependency conflicts between packages.

Examples:
  scripts/run-tests.sh                                   # default: all tests
  scripts/run-tests.sh --all                             # explicit: all tests
  scripts/run-tests.sh --fr                              # FR safety tests only
  scripts/run-tests.sh --cassettes                       # VCR replay only
  scripts/run-tests.sh --cassettes --package "*openai*"  # single package cassettes
  scripts/run-tests.sh --cassettes -- --record-mode=all  # recording mode
  PACKAGE_TEST_TIMEOUT_SECONDS=1800 scripts/run-tests.sh --fr
  scripts/run-tests.sh --fr -- -x
EOF
}

log() {
  printf '[run-tests] %s\n' "$*"
}

die() {
  printf '[run-tests] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

state_file() {
  local package_name="$1"
  local field="$2"
  local package_key
  package_key="$(printf '%s' "$package_name" | tr '/ ' '__')"
  printf '%s/%s.%s' "$STATE_DIR" "$package_key" "$field"
}

set_state() {
  local package_name="$1"
  local field="$2"
  local value="${3:-}"
  printf '%s' "$value" > "$(state_file "$package_name" "$field")"
}

get_state() {
  local package_name="$1"
  local field="$2"
  local file_path
  file_path="$(state_file "$package_name" "$field")"
  if [[ -f "$file_path" ]]; then
    cat "$file_path"
  fi
}

mark_package_seen() {
  local package_name="$1"
  local seen
  for seen in "${PACKAGE_NAMES[@]:-}"; do
    if [[ "$seen" == "$package_name" ]]; then
      return 0
    fi
  done
  PACKAGE_NAMES+=("$package_name")
}

discover_packages() {
  find "$PACKAGES_DIR" -mindepth 2 -maxdepth 2 -type f -name pyproject.toml -print \
    | sed 's#/pyproject.toml$##' \
    | grep -v '/sample-app$' \
    | sort
}

matches_package_filter() {
  local package_dir="$1"
  local package_name
  package_name="$(basename "$package_dir")"

  if [[ -z "$PACKAGE_FILTER" ]]; then
    return 0
  fi

  [[ "$package_name" == $PACKAGE_FILTER ]]
}

has_tests_directory() {
  local package_dir="$1"
  [[ -d "$package_dir/tests" ]]
}

has_marker_in_tests() {
  local package_dir="$1"
  local marker="$2"

  if command -v rg >/dev/null 2>&1; then
    rg -l "pytest\\.mark\\.${marker}" "$package_dir/tests" >/dev/null 2>&1
    return $?
  fi

  grep -RIl -E "pytest\\.mark\\.${marker}" "$package_dir/tests" >/dev/null 2>&1
}

has_fr_source_files() {
  # Check if a package contains FR-authored source files (safety.py,
  # streaming_safety.py, etc.) that indicate it has FR modifications.
  local package_dir="$1"
  find "$package_dir" -path '*/tests' -prune -o \
    \( -name 'safety.py' -o -name 'streaming_safety.py' -o -name 'safety_registration.py' \) \
    -print -quit 2>/dev/null | grep -q .
}

validate_fr_test_coverage() {
  # In --fr mode, warn about packages that have FR source files but no
  # FR-marked tests.  This catches cases where new safety code was added
  # to a package but the developer forgot to add @pytest.mark.fr tests.
  local -a gap_packages=()
  local package_dir

  for package_dir in "$@"; do
    if has_fr_source_files "$package_dir"; then
      if ! has_tests_directory "$package_dir" || ! has_marker_in_tests "$package_dir" "fr"; then
        gap_packages+=("$(basename "$package_dir")")
      fi
    fi
  done

  if (( ${#gap_packages[@]} > 0 )); then
    printf '\n'
    log "WARNING: The following packages have FR safety source files but NO @pytest.mark.fr tests:"
    for pkg in "${gap_packages[@]}"; do
      log "  - $pkg"
    done
    log "Add FR-marked tests to these packages to ensure safety code is covered."
    printf '\n'
  fi
}

python_meta() {
  local package_dir="$1"
  local key="$2"

  python3 - "$package_dir" "$key" <<'PY'
import pathlib
import sys
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        # Minimal TOML parser for just the [project] name field.
        import re
        text = (pathlib.Path(sys.argv[1]).resolve() / "pyproject.toml").read_text()
        if sys.argv[2] == "name":
            m = re.search(r'^name\s*=\s*"([^"]+)"', text, re.M)
            print(m.group(1) if m else pathlib.Path(sys.argv[1]).name)
        raise SystemExit(0)

package_dir = pathlib.Path(sys.argv[1]).resolve()
key = sys.argv[2]
data = tomllib.loads((package_dir / "pyproject.toml").read_text())
project = data.get("project", {})

if key == "name":
    print(project.get("name", package_dir.name))
PY
}

sync_package() {
  local package_dir="$1"
  local package_name="$2"

  local install_log="$REPORT_DIR/${package_name}.install.log"

  # Detect available groups/extras directly from pyproject.toml using grep
  # (no Python dependency — avoids tomllib/tomli availability issues).
  local pyproject="$package_dir/pyproject.toml"
  local -a uv_args=(uv sync)
  if [[ -n "$PYTHON_VERSION" ]]; then
    uv_args+=(--python "$PYTHON_VERSION")
  fi
  if grep -q '^\[dependency-groups\]' "$pyproject" 2>/dev/null && \
     grep -q '^test\s*=' "$pyproject" 2>/dev/null; then
    uv_args+=(--group test)
  fi
  if grep -q 'instruments\s*=' "$pyproject" 2>/dev/null; then
    uv_args+=(--extra instruments)
  fi

  log "Syncing dependencies for $package_name"
  set +e
  (
    cd "$package_dir"
    # Always start with a fresh venv; keep the committed uv.lock to preserve
    # version pins that tests depend on.  If uv sync fails (e.g. stale lock
    # format), retry with a deleted lock file for fresh resolution.
    rm -rf .venv
    if ! "${uv_args[@]}" 2>&1; then
      rm -f uv.lock
      "${uv_args[@]}"
    fi
  ) > >(tee "$install_log") 2>&1
  local sync_status=$?
  set -e

  if [[ $sync_status -ne 0 ]]; then
    # Check if this is a known platform/version build failure.  If so,
    # treat as SKIP instead of INSTALL_FAIL (expected, not actionable).
    local build_skip_reason
    build_skip_reason="$(skip_reason "$(basename "$package_dir")")"
    if [[ -z "$build_skip_reason" ]]; then
      # Not a known skip — detect common patterns from the install log.
      if grep -q "doesn't have a source distribution or wheel for the current platform" "$install_log" 2>/dev/null; then
        build_skip_reason="No compatible wheel for current platform"
      elif grep -q "failed with exit code" "$install_log" 2>/dev/null && grep -q "watchdog\|lancedb\|torch" "$install_log" 2>/dev/null; then
        build_skip_reason="Native dependency build failed (platform-specific)"
      fi
    fi
    if [[ -n "$build_skip_reason" ]]; then
      set_state "$package_name" "status" "SKIP"
      set_state "$package_name" "reason" "$build_skip_reason"
    else
      set_state "$package_name" "status" "INSTALL_FAIL"
      set_state "$package_name" "reason" "uv sync failed. See $install_log"
    fi
    return 1
  fi
}

parse_junit_summary() {
  local junit_xml="$1"

  python3 - "$junit_xml" <<'PY'
import json
import pathlib
import sys
import xml.etree.ElementTree as ET

xml_path = pathlib.Path(sys.argv[1])
summary = {
    "tests": 0,
    "passed": 0,
    "failed": 0,
    "errors": 0,
    "skipped": 0,
    "failing_tests": [],
}

if not xml_path.exists():
    print(json.dumps(summary))
    raise SystemExit(0)

root = ET.parse(xml_path).getroot()
tests = list(root.iter("testcase"))
summary["tests"] = len(tests)

for case in tests:
    is_failed = case.find("failure") is not None
    is_error = case.find("error") is not None
    is_skipped = case.find("skipped") is not None
    if is_failed or is_error:
        summary["failed"] += int(is_failed)
        summary["errors"] += int(is_error)
        classname = case.attrib.get("classname", "").strip()
        name = case.attrib.get("name", "").strip()
        if classname and name:
            summary["failing_tests"].append(f"{classname}::{name}")
        else:
            summary["failing_tests"].append(name or classname or "<unknown>")
    elif is_skipped:
        summary["skipped"] += 1

summary["passed"] = summary["tests"] - summary["failed"] - summary["errors"] - summary["skipped"]
print(json.dumps(summary))
PY
}

run_with_timeout() {
  local timeout_seconds="$1"
  shift

  python3 - "$timeout_seconds" "$@" <<'PY'
import subprocess
import sys

timeout_seconds = int(sys.argv[1])
command = sys.argv[2:]

try:
    completed = subprocess.run(command, timeout=timeout_seconds)
except subprocess.TimeoutExpired:
    print(
        f"[run-tests] ERROR: command timed out after {timeout_seconds} seconds",
        file=sys.stderr,
    )
    raise SystemExit(124)

raise SystemExit(completed.returncode)
PY
}

record_test_summary() {
  local package_name="$1"
  local junit_xml="$2"

  local summary_json
  summary_json="$(parse_junit_summary "$junit_xml")"

  set_state "$package_name" "counts" "$(python3 - "$summary_json" <<'PY'
import json
import sys
data = json.loads(sys.argv[1])
print(f"passed={data['passed']} failed={data['failed']} errors={data['errors']} skipped={data['skipped']} total={data['tests']}")
PY
)"

  set_state "$package_name" "failures" "$(python3 - "$summary_json" <<'PY'
import json
import sys
data = json.loads(sys.argv[1])
print("\n".join(data["failing_tests"]))
PY
)"
}

run_package_tests() {
  local package_dir="$1"
  local package_name
  package_name="$(python_meta "$package_dir" name)"
  mark_package_seen "$package_name"

  if ! has_tests_directory "$package_dir"; then
    set_state "$package_name" "status" "SKIP"
    set_state "$package_name" "reason" "No tests directory"
    return 0
  fi

  local package_basename
  package_basename="$(basename "$package_dir")"
  local pkg_skip_reason
  pkg_skip_reason="$(skip_reason "$package_basename")"
  if [[ -n "$pkg_skip_reason" ]]; then
    set_state "$package_name" "status" "SKIP"
    set_state "$package_name" "reason" "$pkg_skip_reason"
    return 0
  fi

  if [[ "$MODE" == "fr" ]] && ! has_marker_in_tests "$package_dir" "fr"; then
    set_state "$package_name" "status" "SKIP"
    set_state "$package_name" "reason" "No FR-marked tests detected"
    return 0
  fi

  if [[ "$MODE" == "cassettes" ]] && ! has_marker_in_tests "$package_dir" "vcr"; then
    set_state "$package_name" "status" "SKIP"
    set_state "$package_name" "reason" "No VCR-marked tests detected"
    return 0
  fi

  if ! sync_package "$package_dir" "$package_name"; then
    return 0
  fi

  local junit_xml="$REPORT_DIR/${package_name}.xml"
  local test_log="$REPORT_DIR/${package_name}.test.log"
  local -a pytest_cmd
  pytest_cmd=(uv run)
  if [[ -n "$PYTHON_VERSION" ]]; then
    pytest_cmd+=(--python "$PYTHON_VERSION")
  fi
  pytest_cmd+=(pytest -q --junitxml "$junit_xml" tests)
  if [[ "$MODE" == "fr" ]]; then
    pytest_cmd+=(-m fr)
  elif [[ "$MODE" == "cassettes" ]]; then
    pytest_cmd+=(-m vcr)
    # Default to replay-only unless the caller overrides via -- args.
    local has_record_mode=0
    for arg in "${PYTEST_ARGS[@]:-}"; do
      if [[ "$arg" == --record-mode* || "$arg" == --record-mode=* ]]; then
        has_record_mode=1
        break
      fi
    done
    if [[ $has_record_mode -eq 0 ]]; then
      pytest_cmd+=(--record-mode=none)
    fi
  fi
  pytest_cmd+=("${PYTEST_ARGS[@]:-}")

  log "Running tests for $package_name (timeout=${PACKAGE_TEST_TIMEOUT_SECONDS}s)"
  set +e
  (
    cd "$package_dir"
    run_with_timeout "$PACKAGE_TEST_TIMEOUT_SECONDS" "${pytest_cmd[@]}"
  ) > >(tee "$test_log") 2>&1
  local test_status=$?
  set -e

  if [[ $test_status -eq 5 ]]; then
    set_state "$package_name" "status" "SKIP"
    set_state "$package_name" "reason" "Pytest collected no matching tests"
    return 0
  fi

  record_test_summary "$package_name" "$junit_xml"

  if [[ $test_status -eq 0 ]]; then
    set_state "$package_name" "status" "PASS"
  elif [[ $test_status -eq 124 ]]; then
    set_state "$package_name" "status" "FAIL"
    set_state "$package_name" "reason" "Timed out after ${PACKAGE_TEST_TIMEOUT_SECONDS}s. See $test_log"
  else
    set_state "$package_name" "status" "FAIL"
    set_state "$package_name" "reason" "Test failures detected. See $test_log"
  fi
}

print_report() {
  local total=0
  local passed=0
  local failed=0
  local skipped=0
  local install_failed=0
  local tests_total=0
  local tests_passed=0
  local tests_failed=0
  local tests_skipped=0
  local package_name
  local package_counts
  local counts_field
  local counts_value
  local package_passed
  local package_failed
  local package_errors
  local package_skipped
  local package_total
  local -a failing_lines=()
  local -a skipped_lines=()
  local -a executed_lines=()

  printf '\n=== Consolidated Test Report ===\n'
  printf 'Mode: %s\n' "$MODE"
  if [[ -n "$PYTHON_VERSION" ]]; then
    printf 'Python: %s\n' "$PYTHON_VERSION"
  fi
  printf 'Reports: %s\n\n' "$REPORT_DIR"

  for package_name in "${PACKAGE_NAMES[@]:-}"; do
    total=$((total + 1))
    case "$(get_state "$package_name" "status")" in
      PASS)
        passed=$((passed + 1))
        package_counts="$(get_state "$package_name" "counts")"
        executed_lines+=("$(printf 'PASS  %-50s %s' "$package_name" "$package_counts")")
        package_passed=0
        package_failed=0
        package_errors=0
        package_skipped=0
        package_total=0
        for counts_field in $package_counts; do
          counts_value="${counts_field#*=}"
          case "$counts_field" in
            passed=*) package_passed="$counts_value" ;;
            failed=*) package_failed="$counts_value" ;;
            errors=*) package_errors="$counts_value" ;;
            skipped=*) package_skipped="$counts_value" ;;
            total=*) package_total="$counts_value" ;;
          esac
        done
        tests_total=$((tests_total + package_total))
        tests_passed=$((tests_passed + package_passed))
        tests_failed=$((tests_failed + package_failed + package_errors))
        tests_skipped=$((tests_skipped + package_skipped))
        ;;
      FAIL)
        failed=$((failed + 1))
        package_counts="$(get_state "$package_name" "counts")"
        executed_lines+=("$(printf 'FAIL  %-50s %s' "$package_name" "$package_counts")")
        package_passed=0
        package_failed=0
        package_errors=0
        package_skipped=0
        package_total=0
        for counts_field in $package_counts; do
          counts_value="${counts_field#*=}"
          case "$counts_field" in
            passed=*) package_passed="$counts_value" ;;
            failed=*) package_failed="$counts_value" ;;
            errors=*) package_errors="$counts_value" ;;
            skipped=*) package_skipped="$counts_value" ;;
            total=*) package_total="$counts_value" ;;
          esac
        done
        tests_total=$((tests_total + package_total))
        tests_passed=$((tests_passed + package_passed))
        tests_failed=$((tests_failed + package_failed + package_errors))
        tests_skipped=$((tests_skipped + package_skipped))
        if [[ -n "$(get_state "$package_name" "failures")" ]]; then
          while IFS= read -r failing_test; do
            [[ -n "$failing_test" ]] || continue
            failing_lines+=("$package_name :: $failing_test")
          done <<< "$(get_state "$package_name" "failures")"
        fi
        ;;
      INSTALL_FAIL)
        install_failed=$((install_failed + 1))
        executed_lines+=("$(printf 'ERROR %-50s %s' "$package_name" "$(get_state "$package_name" "reason")")")
        failing_lines+=("$package_name :: [install] $(get_state "$package_name" "reason")")
        ;;
      *)
        skipped=$((skipped + 1))
        skipped_lines+=("$(printf 'SKIP  %-50s %s' "$package_name" "$(get_state "$package_name" "reason")")")
        ;;
    esac
  done

  printf 'Skipped packages:\n'
  if (( ${#skipped_lines[@]} > 0 )); then
    printf '%s\n' "${skipped_lines[@]}"
  else
    printf ' - none\n'
  fi

  printf '\nExecuted packages:\n'
  if (( ${#executed_lines[@]} > 0 )); then
    printf '%s\n' "${executed_lines[@]}"
  else
    printf ' - none\n'
  fi

  printf '\nOverall tests run: total = %d passed = %d failed = %d skipped = %d\n' \
    "$tests_total" "$tests_passed" "$tests_failed" "$tests_skipped"

  printf '\nPackages: total=%d passed=%d failed=%d install_failed=%d skipped=%d\n' \
    "$total" "$passed" "$failed" "$install_failed" "$skipped"

  if [[ ${#failing_lines[@]} -gt 0 ]]; then
    printf '\nFailing tests:\n'
    printf ' - %s\n' "${failing_lines[@]}"
  else
    printf '\nFailing tests:\n'
    printf ' - none\n'
  fi

  printf '\n'

  if (( failed > 0 || install_failed > 0 )); then
    return 1
  fi
  return 0
}

main() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --fr)
        MODE="fr"
        shift
        ;;
      --cassettes)
        MODE="cassettes"
        shift
        ;;
      --all)
        MODE="all"
        shift
        ;;
      --package)
        [[ $# -ge 2 ]] || die "--package requires a glob argument"
        PACKAGE_FILTER="$2"
        shift 2
        ;;
      --python)
        [[ $# -ge 2 ]] || die "--python requires a version argument (e.g. 3.12)"
        PYTHON_VERSION="$2"
        shift 2
        ;;
      --list)
        LIST_ONLY=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        PYTEST_ARGS=("$@")
        break
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done

  require_cmd uv
  require_cmd python3

  # Guard: abort if there are locally modified uv.lock files.  The test run
  # re-locks each package (to upgrade the lock format) and restores the
  # committed state afterwards.  Uncommitted changes would be lost.
  if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    local dirty_locks
    dirty_locks="$(git diff --name-only -- '*/uv.lock' 'uv.lock' 2>/dev/null)"
    if [[ -n "$dirty_locks" ]]; then
      die "Uncommitted uv.lock changes detected. Commit or stash them first — the test run modifies and restores uv.lock files.
$dirty_locks"
    fi
  fi

  mkdir -p "$REPORT_DIR"
  mkdir -p "$STATE_DIR"

  local -a package_dirs=()
  local package_dir
  while IFS= read -r package_dir; do
    [[ -n "$package_dir" ]] || continue
    if matches_package_filter "$package_dir"; then
      package_dirs+=("$package_dir")
    fi
  done < <(discover_packages)

  (( ${#package_dirs[@]} > 0 )) || die "No packages matched the requested filters"

  if (( LIST_ONLY == 1 )); then
    printf '%s\n' "${package_dirs[@]}"
    exit 0
  fi

  # In --fr mode, validate that every package with FR source files also
  # has FR-marked tests.  This runs before test execution so gaps are
  # visible even if the test run itself is aborted.
  if [[ "$MODE" == "fr" ]]; then
    validate_fr_test_coverage "${package_dirs[@]}"
  fi

  local package_name
  for package_dir in "${package_dirs[@]}"; do
    package_name="$(python_meta "$package_dir" name)"
    mark_package_seen "$package_name"
    set_state "$package_name" "status" "SKIP"
    set_state "$package_name" "reason" "Not executed"
  done

  for package_dir in "${package_dirs[@]}"; do
    run_package_tests "$package_dir"
  done

  # Restore any uv.lock files that uv sync may have updated (format upgrade).
  if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git checkout -- '*/uv.lock' 2>/dev/null || true
  fi

  print_report
}

main "$@"
