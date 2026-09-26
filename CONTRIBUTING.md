# Contributing to bibr 🦫

Thank you for your interest in contributing to bibr! This guide covers the development setup, testing, and contribution workflow.

## Development setup

```bash
# Clone the repository
git clone https://github.com/scienceverse/bibr.git
cd bibr

# Install development/docs tools and the cloud + ML dependency set
# `--extra all` selects the cloud + ML superset; add the serving extras
# (local, vllm, local-mlx, gpu) that match your hardware.
uv sync --locked --extra all --all-groups

# Configure .env when you want to run live extraction
uv run bibr setup
```

### Prerequisites

- Python 3.11–3.14 (3.12 is the primary CI version; optional serving runtimes
  may require a narrower version range)
- [uv](https://docs.astral.sh/uv/) package manager
- `libmagic` on Linux/macOS; see [Installation](https://bibr.org/getting-started/install/)
- For live extraction: an OCR backend and a configured cloud or local LLM.
  The `local` extra installs Apple Silicon serving dependencies; it is not
  a universal OCR installation. Use the [setup wizard and Quickstart](https://bibr.org/getting-started/quickstart/)
  to select runtimes for your hardware.

Unit tests do not require real API keys: `tests/conftest.py` supplies placeholder
settings and tests mock external calls. For a core-only or docs environment, use
`uv sync --locked --group docs` and let optional-dependency tests skip.

## Running tests

```bash
# Run tests (skipping slow integration tests, matching CI)
uv run --locked --extra all pytest -m "not slow"

# Run a single test file
uv run pytest tests/test_config.py

# Run a specific test with verbose output
uv run pytest tests/test_config.py::test_settings_default -v

# Run with coverage
uv run pytest -m "not slow" --cov=bibr
```

CI runs the non-slow suite in both full and core-only environments. Slow tests
are excluded by the `-m` expression; bare `pytest` includes them. Depending on
the test, they may require model downloads, GPU hardware, or live OCR/LLM
services. Check their individual requirements before running them.

## Linting and formatting

bibr uses [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting:

```bash
# Check for lint errors
uv run ruff check .

# Auto-fix lint errors
uv run ruff check . --fix

# Format code
uv run ruff format .

# Check formatting without changing files (as CI does)
uv run ruff format --check .
```

### Code style

- Line length: 100 characters
- Quotes: double quotes
- Target: Python 3.11 (`py311`)
- Rules: E/W/F/I/B/C4/UP/ARG/SIM/S/LOG
- `assert` statements allowed in tests (`S101` ignored)

## Pre-commit hooks

The repository uses pre-commit hooks for automated checks:

- **Ruff** (lint + format), run through the project venv so it matches CI exactly
- **Semgrep** (`p/security-audit`, `p/secrets`, `p/python` rulesets) — the same static-analysis packs CI runs
- Standard file checks: `check-yaml`, `check-json`, `check-toml`, `check-added-large-files`, `check-merge-conflict`, `detect-private-key`

Install hooks:

```bash
uv run pre-commit install
```

## Code conventions

- **Public API**: use `bibr.chew()` / `bibr.achew()` for one-off extraction and
  `bibr.Chewer` for a reusable session. The typed `chew_file()` / `chew_many()`
  helpers and their async equivalents provide fixed single/batch return types.
  `Result.model` exposes the validated `PaperExport`; batch failures retain
  their input position as `ChewFailure`. `LocalPipeline`, `Pipeline`,
  `GlobalSettings`, and `Settings` remain available for lower-level control.
  Top-level exports use lazy `__getattr__` so imports stay light.
- **Lazy `__getattr__` imports** in `__init__.py` files for subpackages with heavy dependencies (transformers, torch). Follow this pattern when adding new subpackages.
- **Settings**: `bibr/config.py` lazily exposes `GlobalSettings` through `Settings`, with sub-models accessed as `Settings.ocr.backend`, `Settings.llm.provider`, etc. Explicit settings can be passed to pipelines for isolated configurations. Reads from `.env` or environment variables.
- **Exception hierarchy**: `BibrError` base class with `InputValidationError`, `UpstreamServiceError`, `ProcessingError`.
- **Async pipeline**: `Pipeline.process_chunk()` uses `asyncio.gather` with a semaphore for batch concurrency. Tests use `pytest-asyncio` with `asyncio_mode = "auto"`.
- **Module singletons**: Lazy-initialized with `_ensure_initialized()` and `threading.Lock()` for thread safety.

## Architecture overview

bibr processes papers through six pipeline stages:

1. **Validate** -- MIME type and corruption checks
2. **Input** -- native DOCX, XML/JATS, HTML, and ePub parsing; bounded PDF page windows combine layout detection, embedded-text reconstruction, and OCR for regions that need it
3. **Structure** -- section classification, sentence segmentation, citation linking
4. **Extract** -- LLM metadata extraction plus configured local classifiers and independent reference segmentation/parsing strategies
5. **Enrich** -- optional Crossref reference enrichment
6. **Export** -- JSON serialization (versioned schema -- see the [JSON schema reference](https://bibr.org/reference/schema/))

See the [Architecture docs](https://bibr.org/guides/architecture/) for details.

## Pull request guidelines

1. Branch from `main`; use descriptive branch names without `codex/` or `claude/` prefixes
2. Write tests for new functionality
3. Run the relevant non-slow tests, Ruff checks, and strict docs build before submitting
4. Use descriptive commit messages without `Co-authored-by` trailers
5. Keep PRs focused on a single concern

## Documentation

Update documentation with changes to CLI options, defaults, formats, schemas,
or deployment behavior. Preview and check the site with:

```bash
uv run --locked --group docs mkdocs serve
uv run --locked --group docs mkdocs build --strict
uv run --locked pytest tests/test_docs_sync.py
```

Settings, CLI, and JSON-schema reference pages are generated from live code by
`scripts/docs_ref_core.py` during the build. Edit their source descriptions or
generator rather than built HTML. Add user-facing pages to `mkdocs.yml` and
keep linked pages consistent. The build permits only navigated pages and
reviewed static assets; do not add paper corpora, extraction captures, model
experiments, or local filesystem paths to the documentation. See the [development guide](https://bibr.org/contributing/setup/)
for details and the [evaluation guide](https://bibr.org/contributing/evaluation/)
for extraction-quality measurements.

Maintainers can follow the [release publication guide](https://bibr.org/contributing/setup/#publishing-a-release)
for rehearsals, Trusted Publishing, and recovery from a partial release.

## Testing conventions

- Use `pytest.importorskip()` for optional dependencies (`fastapi`, `redis`)
- `tests/conftest.py` sets environment variables at module level so `GlobalSettings` doesn't fail
- Mark integration tests with `@pytest.mark.slow`
- Mock heavy models in unit tests; use `sys.modules` patching where needed
- Keep unit tests offline: the autouse socket guard fails any test that opens
  a non-loopback socket or resolves an external name. Tests that intentionally
  reach the network (live API, Hub downloads) take `@pytest.mark.network`
- Global `Settings` mutations and `bibr.*` logger levels are restored after
  every test by autouse fixtures — no manual `model_fields_set` juggling
- The torch-free ONNX bundles under `tests/fixtures/onnx` and the hermetic
  LitServe/geometry fixtures are committed artifacts: regenerate them with
  `scripts/generate_onnx_test_bundles.py` and
  `scripts/generate_hermetic_test_fixtures.py` instead of editing by hand

## Contributor License Agreement

A CLA must be signed before your first PR can be merged. The maintainer will provide the CLA text on your first pull request; this process is still being finalized ahead of wider release.

## License

By contributing, you agree that your contributions will be licensed under the project's [AGPL-3.0-or-later](LICENSE.md) license.
