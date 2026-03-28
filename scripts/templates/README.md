# Test Templates

Templates for standardizing test infrastructure across `fr-openllmetry-py` packages.

## Files

| Template | Purpose |
|----------|---------|
| `conftest_vcr.py` | Standard conftest.py for packages with VCR cassette tests. Includes OTel exporter setup, VCR config with secret filtering, and documented extension points. |

## Usage

```bash
# Copy the VCR conftest template to a new package
cp scripts/templates/conftest_vcr.py packages/<your-package>/tests/conftest.py

# Then edit the TODO sections:
# 1. PROVIDER_ENV_VARS — dummy API keys for cassette replay
# 2. PROVIDER_FILTER_HEADERS — headers to strip from cassettes
# 3. Provider client fixtures — SDK client creation
# 4. Instrumentor fixtures — OTel instrumentation setup
```

## Conventions

- `vcr_config` fixture is `module`-scoped (shared across tests in a file)
- `span_exporter` / `tracer_provider` are `function`-scoped (fresh per test)
- `environment` fixture sets dummy API keys (autouse) so replay works without real keys
- `clear_exporter` fixture (autouse) clears spans before each test
- Cassettes stored in `tests/cassettes/` (or subdirectory like `tests/traces/cassettes/`)
- Secret filtering via `filter_headers` and `filter_query_parameters` is mandatory
