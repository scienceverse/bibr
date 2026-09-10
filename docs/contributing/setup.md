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
`mcp`, `torch`). The hardware-specific serving extras (`local`, `vllm`,
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

## Publishing a release

The release workflow builds the wheel and source archive once, checks their
contents and PyPI metadata, and installs the wheel in a clean environment. It
publishes those same files through PyPI Trusted Publishing with attestations.
The package description comes from `pyproject.toml`; the PyPI project page
comes from `README.md`. Update both before releasing, including installation
instructions and links that work outside GitHub.

The registered PyPI publisher must match these values:

| Setting | Value |
| --- | --- |
| PyPI project | `bibr` |
| GitHub owner/repository | `scienceverse/bibr` |
| Workflow filename | `release.yml` |
| GitHub environment | `pypi` |

The `pypi` environment allows `v*` tags. No PyPI API token is needed. Keep the
repository variable `PUBLISH_PYPI=false` between releases; a release owner enables
it after approving publication.

Container registry delivery is a separate opt-in: `PUBLISH_GHCR=true` enables
edge and release uploads, including manual container workflow runs. It is
disabled for the initial public 0.5.0 launch; users can build the containers
from the public source. CI still builds the serve image and blocks fixable
HIGH/CRITICAL vulnerabilities before `CI / required` passes.

Before enabling GHCR, configure a clean package with the intended visibility
and repository Actions access, then verify anonymous pulls for public images.
Do not expose a legacy private package's old versions as part of that setup.
See [Docker deployment](../guides/deployment.md#docker-deployment).

1. Update the package version, lockfile, changelog, and public documentation on
   `main`. Wait for `CI / required` to pass on the exact commit to be released.
2. Rehearse the release from `main` with
   `gh workflow run release.yml --repo scienceverse/bibr --ref main`. Wait for
   the Ubuntu, macOS, Windows, and distribution checks to pass. A manual rehearsal
   cannot publish to PyPI, GHCR, or GitHub Releases.
3. After release approval, set
   `gh variable set PUBLISH_PYPI --repo scienceverse/bibr --body true`.
   Create and push an annotated `vX.Y.Z` tag on the verified commit, with `X.Y.Z`
   matching `project.version`. The workflow rejects mismatched tags and commits
   that are not reachable from `main`.
4. Watch the tag-triggered Release workflow to completion. PyPI receives the
   verified distributions and GitHub Release assets are attached after each
   enabled delivery channel succeeds. If GHCR is enabled, the release container
   must also pass its digest scan before its version tags are promoted. A failed
   enabled channel blocks finalization; only deliberately disabled channels may
   be skipped.
5. Verify the live PyPI description and install that exact version from PyPI in
   a fresh environment. Then reset
   `gh variable set PUBLISH_PYPI --repo scienceverse/bibr --body false`.

If publication partially succeeds, rerun only the failed jobs of that same run.
Do not rerun a successful PyPI upload or move a published release tag. PyPI files
are immutable; package or description corrections require a new version.

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
