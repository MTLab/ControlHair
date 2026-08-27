#!/usr/bin/env python3
"""Prepare public and separately licensed model assets from official sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "model_sources.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_receipt(destination: Path, config: dict[str, Any], files: list[Path]) -> None:
    receipt_dir = destination if destination.is_dir() else destination.parent
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "source": config,
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
            if path.is_file()
        ],
    }
    (receipt_dir / ".controlhair-prepared.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_huggingface(config: dict[str, Any], dry_run: bool) -> None:
    destination = ROOT / config["destination"]
    print(
        f"Hugging Face: {config['repository']}@{config['revision']} -> "
        f"{destination.relative_to(ROOT)}"
    )
    if config.get("download_patterns"):
        print("Selected files: " + ", ".join(config["download_patterns"]))
    if dry_run:
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install requirements-release.txt before downloading HF models") from exc
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=config["repository"],
        revision=config["revision"],
        local_dir=destination,
        token=True if config.get("private") else None,
        allow_patterns=config.get("download_patterns"),
    )
    files = [
        destination / pattern
        for pattern in config.get("download_patterns", [])
        if (destination / pattern).is_file()
    ]
    write_receipt(destination, config, files)


def prepare_hairstep_networks(config: dict[str, Any], dry_run: bool) -> None:
    checkout = ROOT / "third_party" / "HairStep"
    destination = ROOT / config["destination"]
    required = ROOT / config["required_file"]
    print(f"HairStep restricted checkpoint archive -> {destination.relative_to(ROOT)}")
    if dry_run:
        return
    if not (checkout / ".git").is_dir():
        raise RuntimeError(
            "Prepare the HairStep source first with scripts/prepare_external_dependencies.py"
        )
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Install requirements-release.txt before downloading HairStep assets") from exc
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "hairstep_networks.download"
    gdown.download(id=config["file_id"], output=str(archive), quiet=False)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            destination_root = destination.resolve()
            for member in handle.infolist():
                member_path = (destination / member.filename).resolve()
                if destination_root not in member_path.parents and member_path != destination_root:
                    raise RuntimeError(f"Unsafe path in HairStep archive: {member.filename}")
            handle.extractall(destination)
        archive.unlink()
    if not required.is_file():
        raise RuntimeError(
            f"HairStep archive was downloaded, but {required.relative_to(ROOT)} is missing. "
            "Follow the upstream README and place the authorized img2strand checkpoint there."
        )
    write_receipt(destination, config, [required])


def prepare_controlhair_override(config: dict[str, Any], checkpoint: Path, dry_run: bool) -> None:
    destination = ROOT / config["destination"]
    print(f"ControlHair checkpoint: {checkpoint} -> {destination.relative_to(ROOT)}")
    if dry_run:
        return
    if not checkpoint.is_file() and not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / checkpoint.name
    if checkpoint.is_dir():
        shutil.copytree(checkpoint, target, dirs_exist_ok=True)
        files = sorted(path for path in target.rglob("*") if path.is_file())
    else:
        shutil.copy2(checkpoint, target)
        files = [target]
    write_receipt(destination, config, files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component",
        action="append",
        choices=["wan", "unianimate", "hairstep_networks", "controlhair"],
        required=True,
    )
    parser.add_argument("--accept-third-party-licenses", action="store_true")
    parser.add_argument("--controlhair-checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    models = manifest["models"]
    selected = list(dict.fromkeys(args.component))

    restricted = [name for name in selected if models[name].get("restricted")]
    if restricted and not args.accept_third_party_licenses:
        raise SystemExit(
            "Restricted components selected: "
            + ", ".join(restricted)
            + ". Review THIRD_PARTY_NOTICES.md and rerun with "
            "--accept-third-party-licenses."
        )

    for name in selected:
        config = models[name]
        print(f"\n{name}: {config['license']}")
        kind = config["kind"]
        if name == "controlhair" and args.controlhair_checkpoint is not None:
            prepare_controlhair_override(
                config, args.controlhair_checkpoint.expanduser().resolve(), args.dry_run
            )
        elif kind == "huggingface_snapshot":
            prepare_huggingface(config, args.dry_run)
        elif kind == "google_drive_manual":
            prepare_hairstep_networks(config, args.dry_run)
        else:
            raise RuntimeError(f"Unsupported preparation kind: {kind}")


if __name__ == "__main__":
    main()
