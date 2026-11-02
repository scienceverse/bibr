"""Allow ``python -m bibr.local.cli`` to run the CLI (package entry point)."""

from bibr.local.cli import entrypoint

if __name__ == "__main__":
    entrypoint()
