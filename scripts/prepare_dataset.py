#!/usr/bin/env python3
"""Prepare private ControlHair datasets from pinned Hugging Face revisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "model_sources.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component",
        choices=["controlhair_480p"],
        required=True,
    )
    parser.add_argument("--accept-private-dataset-terms", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    config = manifest["datasets"][args.component]
    destination = ROOT / config["destination"]

    if not args.accept_private_dataset_terms:
        raise SystemExit(
            "Review the dataset terms and rerun with --accept-private-dataset-terms."
        )

    print(
        f"Hugging Face dataset: {config['repository']}@{config['revision']} -> "
        f"{destination.relative_to(ROOT)}"
    )
    print(f"Expected samples: {config['samples']}; bytes: {config['bytes']}")
    if args.dry_run:
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install requirements-release.txt before downloading datasets") from exc

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=config["repository"],
        repo_type="dataset",
        revision=config["revision"],
        local_dir=destination,
        token=True,
    )
    receipt = {
        "source": config,
        "prepared_revision": config["revision"],
        "note": "Private dataset; access and redistribution remain governed by the repository owner.",
    }
    (destination / ".controlhair-prepared.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
