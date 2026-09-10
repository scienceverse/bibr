# Development setup

## Clone and install

Use Python 3.11–3.14 and [uv](https://docs.astral.sh/uv/). Python 3.12 is
the primary CI environment; some optional model-serving runtimes have narrower
Python and platform support. Install `libmagic` on Linux/macOS if your system
does not already provide it (see [Installation](../getting-started/install.md)).

```bash
git clone https://github.com/scienceverse/bibr.git
cd bibr

uv sync --locked --extra all --all-groups
```

`--extra all` selects the cloud + ML superset (`batch`, `cache`, `demo`,
`mcp`, `ml`). The hardware-specific serving extras (`local`, `vllm`,
`local-mlx`, `gpu`) are opt-in; `--all-extras` also resolves if you want
every one, subject to their platform markers. `local` is an Apple Silicon
serving extra; it does not install a universal in-process OCR engine.

For core code or documentation work without local ML models:

```bash
uv sync --locked --group docs
```

Tests set placeholder configuration in `tests/conftest.py` and mock external
services. API keys and model downloads are needed only for the live workflows
you choose to run. Use `uv run bibr setup` to configure one of those workflows.

## Running tests

```bash
uv run --locked --extra all pytest -m "not slow"
```

This selects the main CI suite. Core-only CI also runs the non-slow suite with
`uv run --locked pytest -m "not slow"`; tests that need absent optional
dependencies should use `pytest.importorskip()`. Slow tests are excluded by
the `-m` expression, not by pytest's default configuration. Read a slow test's
requirements before enabling it: it may need model downloads, an OCR service,
GPU hardware, or LLM credentials.

## Linting and formatting

```bash
uv run ruff check .
uv run ruff format .
uv run ruff format --check .  # non-mutating CI check
```

bibr uses [Ruff](https://docs.astral.sh/ruff/) for both. Line length is 100
characters, double quotes, Python 3.11+ target.

## Previewing the docs

```bash
uv run --locked --group docs mkdocs serve
uv run --locked --group docs mkdocs build --strict
```

Some reference pages (`reference/settings.md`, `reference/cli.md`,
`reference/schema.md`) are generated at build time from `scripts/docs_ref_core.py`
reading live code and docstrings — don't edit the built output directly. To
change what those pages say, edit `scripts/docs_ref_core.py` or the
docstrings/field descriptions it reads from. The docs build also validates
internal links; add pages to `mkdocs.yml` when they should appear in navigation.
Only navigated pages and reviewed static assets are published. Keep paper
corpora, extraction captures, model experiments, and local filesystem paths out
of documentation sources and generated descriptions.

For the documentation contracts and CLI examples:

```bash
uv run --locked pytest tests/test_docs_sync.py
```

## Conventions

- Line length 100, double quotes, `py311`+ target (see `pyproject.toml` for
  the full Ruff rule set)
- Lazy `__getattr__` imports in `__init__.py` for subpackages with heavy
  dependencies (transformers, torch, etc.), so `import bibr` stays light
- Exceptions derive from `BibrError`; use the appropriate input, processing,
  upstream-service, or configuration error in `bibr/exceptions.py`
- `Settings` in `bibr/config.py` lazily exposes `GlobalSettings`; settings can
  also be passed explicitly to pipeline instances. Read environment variables
  and `.env` through this configuration layer

## Submitting a pull request

- Branch from `main`, keep PRs focused on a single concern, and use descriptive
  branch names without `codex/` or `claude/` prefixes
- Add tests for new functionality
- CI must be green, including the strict docs build (`mkdocs build --strict`)
- Update user-facing docs when CLI options, defaults, input formats, output
  fields, or deployment behavior change
- A CLA is required before your first PR can be merged — see
  [CONTRIBUTING.md](https://github.com/scienceverse/bibr/blob/main/CONTRIBUTING.md)
  for details
