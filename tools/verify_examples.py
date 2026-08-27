#!/usr/bin/env python3
"""Verify example manifests and artifact checksums without media dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("examples"))
    args = parser.parse_args()

    manifests = sorted(args.root.rglob("metadata.json"))
    if not manifests:
        raise SystemExit(f"No metadata.json files found below {args.root}")

    failures: list[str] = []
    for manifest_path in manifests:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 1:
            failures.append(f"{manifest_path}: unsupported schema_version")

        layout = metadata.get("layout", {})
        if layout.get("type") != "equal_width_triptych":
            failures.append(f"{manifest_path}: unexpected layout type")
        if layout.get("source_width", 0) != layout.get("panel_width", -1) * 3:
            failures.append(f"{manifest_path}: invalid triptych dimensions")

        for name, details in metadata.get("artifacts", {}).items():
            artifact_path = manifest_path.parent / details["path"]
            if not artifact_path.is_file():
                failures.append(f"{manifest_path}: missing {name}: {artifact_path.name}")
                continue
            if artifact_path.stat().st_size != details["bytes"]:
                failures.append(f"{manifest_path}: size mismatch for {artifact_path.name}")
            if sha256(artifact_path) != details["sha256"]:
                failures.append(f"{manifest_path}: checksum mismatch for {artifact_path.name}")

        status = metadata.get("license", {}).get("status")
        print(f"OK {manifest_path.parent} license={status}")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        raise SystemExit(1)

    print(f"Verified {len(manifests)} example packages")


if __name__ == "__main__":
    main()
