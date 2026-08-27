#!/usr/bin/env python3
"""Clone separately licensed source dependencies at audited revisions."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "model_sources.json"


def command_text(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in command)


def run(command: list[str], dry_run: bool) -> None:
    print(f"+ {command_text(command)}")
    if not dry_run:
        subprocess.run(command, check=True)


def clone_pinned(name: str, config: dict[str, object], dry_run: bool) -> None:
    destination = ROOT / str(config["destination"])
    repository = str(config["repository"])
    revision = str(config["revision"])

    if destination.exists() and not (destination / ".git").is_dir():
        raise RuntimeError(f"{destination} exists but is not a Git checkout")

    if not destination.exists():
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
        run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", repository, str(destination)],
            dry_run,
        )
    run(["git", "-C", str(destination), "fetch", "origin", revision], dry_run)
    run(["git", "-C", str(destination), "checkout", "--detach", revision], dry_run)

    if not dry_run:
        actual = subprocess.run(
            ["git", "-C", str(destination), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual != revision:
            raise RuntimeError(f"{name}: expected {revision}, got {actual}")
        print(f"Prepared {name} at {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component",
        action="append",
        choices=["hairstep", "difflocks", "all"],
        help="Dependency to prepare; repeat to select multiple",
    )
    parser.add_argument("--accept-restricted-licenses", action="store_true")
    parser.add_argument("--download-difflocks-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = args.component or ["all"]
    if "all" in selected:
        selected = ["hairstep", "difflocks"]
    selected = list(dict.fromkeys(selected))

    if not args.accept_restricted_licenses:
        raise SystemExit(
            "Refusing to prepare restricted dependencies. Review THIRD_PARTY_NOTICES.md "
            "and rerun with --accept-restricted-licenses."
        )

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sources = manifest["source_repositories"]
    for name in selected:
        config = sources[name]
        print(f"\n{name}: {config['license']}")
        if name == "difflocks":
            run(["bash", str(ROOT / "scripts" / "setup_difflocks.sh")], args.dry_run)
        else:
            clone_pinned(name, config, args.dry_run)

    if args.download_difflocks_checkpoints:
        if "difflocks" not in selected:
            raise SystemExit("Select --component difflocks when requesting its checkpoints")
        script = ROOT / manifest["models"]["difflocks_checkpoints"]["script"]
        if args.dry_run:
            run(["bash", str(script)], True)
        else:
            if not script.is_file():
                raise FileNotFoundError(script)
            print(
                "Running the official DiffLocks downloader interactively. Credentials are handled "
                "only by the upstream script and are not stored by ControlHair."
            )
            subprocess.run(["bash", str(script)], cwd=script.parent, check=True)


if __name__ == "__main__":
    main()
