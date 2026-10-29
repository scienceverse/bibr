"""Push the demo to a ZeroGPU Space, test it on a paper, and publish it.

    python space_measure.py launch  [--public] [--name bibr-demo] [--owner NAME]
    python space_measure.py push    [--name bibr-demo] [--owner NAME]
    python space_measure.py measure paper.pdf [--runs 2]
    python space_measure.py publish

Uses the Hugging Face login already on this machine; the token is never
printed. ``push`` creates the Space private, on ZeroGPU, uploads this folder's
Space files in one commit and waits for the build. ``measure`` runs a paper
through the Space and prints the GPU time it took. ``launch`` does both with
the example paper and, with ``--public``, makes the Space public only if the
run returned a title, authors and references. ``publish`` makes it public.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).parent
SPACE_FILES = ["app.py", "zerogpu_models.py", "requirements.txt", "README.md", "examples/*.pdf"]
EXAMPLE = HERE / "examples" / "2506.01266v1.pdf"
DONE_STAGES = {"RUNNING", "BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}


def space_id(api: HfApi, args) -> str:
    owner = args.owner or api.whoami()["name"]
    return f"{owner}/{args.name}"


def tail_logs(repo: str, kind: str, seconds: float = 20, lines: int = 60) -> None:
    """Print the last *lines* of the Space's build or run log."""
    from huggingface_hub import get_token
    from huggingface_hub.utils import get_session

    url = f"https://huggingface.co/api/spaces/{repo}/logs/{kind}"
    got: list[str] = []
    headers = {"Authorization": f"Bearer {get_token()}"}
    session = get_session()
    try:
        # huggingface_hub 1.x and later use an httpx client, older ones requests.
        if hasattr(session, "stream"):
            stream = session.stream("GET", url, headers=headers, timeout=seconds)
        else:
            stream = session.get(url, headers=headers, stream=True, timeout=seconds)
        with stream as resp:
            started = time.monotonic()
            for raw in resp.iter_lines():
                line = raw.decode(errors="replace") if isinstance(raw, bytes) else raw
                if line.startswith("data: "):
                    try:
                        got.append(json.loads(line[6:]).get("data", "").rstrip())
                    except (ValueError, AttributeError):
                        got.append(line[6:])
                if time.monotonic() - started > seconds:
                    break
    except Exception as exc:  # noqa: BLE001 - logs are a diagnostic only
        print(f"(could not read the {kind} log: {type(exc).__name__}: {exc})")
    print(f"--- last {kind} log lines ---")
    print("\n".join(got[-lines:]))


def push(api: HfApi, args) -> str:
    """Create or update the private Space and wait for it; return its stage."""
    from huggingface_hub.utils import HfHubHTTPError

    repo = space_id(api, args)
    try:
        api.create_repo(
            repo, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g",
            private=True, exist_ok=True,
        )
    except HfHubHTTPError as exc:
        sys.exit(f"Hugging Face would not create {repo} on ZeroGPU: {exc}")
    info = api.space_info(repo)
    if not info.private and not args.allow_public_update:
        sys.exit(f"{repo} is already public; rerun with --allow-public-update to update it")
    hardware = api.get_space_runtime(repo).requested_hardware
    if hardware != "zero-a10g":
        try:
            api.request_space_hardware(repo, "zero-a10g")
        except HfHubHTTPError as exc:
            sys.exit(f"{repo} is on {hardware} and could not be moved to ZeroGPU: {exc}")
    # The extractor is a Space variable; the card preloads that model's weights.
    api.add_space_variable(repo, "DEMO_EXTRACTOR", args.extractor)
    card = (HERE / "README.md").read_text()
    card = card.replace("scienceverse/bibr-extract-qwen35-4b", f"scienceverse/bibr-extract-qwen35-{args.extractor}")
    from huggingface_hub import CommitOperationAdd

    ops = [CommitOperationAdd("README.md", card.encode())]
    for pattern in SPACE_FILES:
        for path in sorted(HERE.glob(pattern)):
            if path.name != "README.md":
                ops.append(CommitOperationAdd(path.relative_to(HERE).as_posix(), str(path)))
    api.create_commit(
        repo, operations=ops, repo_type="space",
        commit_message=f"Update the bibr demo ({args.extractor} extractor)",
    )
    print(f"pushed to {repo} ({'private' if info.private else 'PUBLIC'}): https://huggingface.co/spaces/{repo}")
    time.sleep(20)  # let the Space pick up the new commit
    started = time.monotonic()
    while time.monotonic() - started < 3600:
        runtime = api.get_space_runtime(repo)
        print(f"{(time.monotonic() - started) / 60:5.1f} min  stage: {runtime.stage}", flush=True)
        if runtime.stage in DONE_STAGES:
            break
        time.sleep(30)
    if runtime.stage != "RUNNING":
        error = (runtime.raw or {}).get("errorMessage")
        print(f"The Space stopped at {runtime.stage}. {error or ''}")
        tail_logs(repo, "build")
        tail_logs(repo, "run")
    return runtime.stage


def _rows(table) -> int:
    if isinstance(table, dict):
        data = table.get("data")
        if data is None and isinstance(table.get("value"), dict):
            data = table["value"].get("data")
        return len(data or [])
    return len(table or [])


def quota(api: HfApi) -> str:
    try:
        q = api.get_zero_gpu_quota()
        return json.dumps(q if isinstance(q, dict) else vars(q), default=str)
    except Exception as exc:  # noqa: BLE001 - a diagnostic only
        return f"unavailable ({type(exc).__name__})"


def run_paper(api: HfApi, args, pdf: Path) -> bool:
    """Run *pdf* through the Space; True when it returned title, authors and references."""
    from gradio_client import Client, handle_file
    from huggingface_hub import get_token

    repo = space_id(api, args)
    # oauth_token passes the Space's sign-in check; token opens the private Space.
    client = Client(repo, token=get_token(), oauth_token=get_token(), verbose=False)
    print("ZeroGPU quota before:", quota(api))
    started = time.monotonic()
    try:
        summary, authors, refs, checks, _ = client.predict(handle_file(str(pdf)), api_name="/chew")
    except Exception as exc:  # noqa: BLE001 - reported, then the run logs show why
        print(f"The paper failed after {time.monotonic() - started:.0f} s: {type(exc).__name__}: {exc}")
        tail_logs(repo, "run")
        return False
    wall = time.monotonic() - started
    print("ZeroGPU quota after: ", quota(api))
    n_authors, n_refs = _rows(authors), _rows(refs)
    print(f"{pdf.name}: {wall:.0f} s wall, {n_authors} authors, {n_refs} references")
    print(str(summary)[:1500])
    print(str(checks)[:1500])
    ok = n_authors >= 1 and n_refs >= 10 and "#" in str(summary)
    if not ok:
        print("The result looks wrong (no title, no authors, or under 10 references).")
        tail_logs(repo, "run")
    return ok


def status(api: HfApi, args) -> None:
    """Print the Space's stage, its error message, and the end of its logs."""
    repo = space_id(api, args)
    runtime = api.get_space_runtime(repo)
    print(f"{repo}: stage {runtime.stage}, hardware {runtime.hardware}")
    print("error:", (runtime.raw or {}).get("errorMessage"))
    tail_logs(repo, "run", lines=40)


def publish(api: HfApi, args) -> None:
    repo = space_id(api, args)
    try:
        api.update_repo_settings(repo_id=repo, repo_type="space", private=False)
    except AttributeError:  # older huggingface_hub
        api.update_repo_visibility(repo, private=False, repo_type="space")
    print(f"PUBLIC: https://huggingface.co/spaces/{repo}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["launch", "push", "measure", "publish", "status"])
    parser.add_argument("pdf", nargs="?", type=Path, default=EXAMPLE)
    parser.add_argument("--name", default="bibr-demo")
    parser.add_argument("--owner")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--public", action="store_true", help="launch: publish if the test passes")
    parser.add_argument("--allow-public-update", action="store_true")
    parser.add_argument("--extractor", choices=["4b", "2b"], default="4b")
    args = parser.parse_args()
    api = HfApi()
    if args.command == "push":
        push(api, args)
    elif args.command == "measure":
        for _ in range(args.runs):
            run_paper(api, args, args.pdf)
    elif args.command == "publish":
        publish(api, args)
    elif args.command == "status":
        status(api, args)
    else:
        try:  # what the last run of the Space ended with, before replacing it
            status(api, args)
        except Exception as exc:  # noqa: BLE001 - a new Space has no status yet
            print(f"(no earlier status: {type(exc).__name__})")
        if push(api, args) != "RUNNING":
            sys.exit("Not published: the Space did not start.")
        if not run_paper(api, args, args.pdf):
            sys.exit("Not published: the test paper did not come back right. The Space stays private.")
        if args.public:
            publish(api, args)
        else:
            print("Test passed. Publish with: python space_measure.py publish")


if __name__ == "__main__":
    main()
