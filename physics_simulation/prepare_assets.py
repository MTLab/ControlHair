#!/usr/bin/env python3
"""Verify and install the Git LFS ControlHair Blender templates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "physics_simulation" / "assets"
DESTINATION = ROOT / "third_party" / "difflocks" / "inference" / "assets"
ASSETS = {
    "drive": {
        "filename": "lwk_template_drive.blend",
        "bytes": 171_367_993,
        "sha256": "089f0dff771d26e4b0e70bad50b7207617946f83af6f2680ac059986e2f0cfde",
    },
    "wind": {
        "filename": "lwk_template_wind.blend",
        "bytes": 171_390_813,
        "sha256": "089cc16a3fd6cc00700328e06bf49d223ca7d8a9ffd592a0e67907aca6b7da6b",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drive-template", type=Path, default=SOURCE / "lwk_template_drive.blend"
    )
    parser.add_argument(
        "--wind-template", type=Path, default=SOURCE / "lwk_template_wind.blend"
    )
    args = parser.parse_args()
    if not (DESTINATION.parents[1] / ".git").exists():
        raise SystemExit("Prepare DiffLocks first with scripts/setup_physics.sh")

    DESTINATION.mkdir(parents=True, exist_ok=True)
    receipt = {"assets": {}}
    for name, source in (("drive", args.drive_template), ("wind", args.wind_template)):
        source = source.expanduser().resolve()
        expected = ASSETS[name]
        if source.stat().st_size != expected["bytes"] or sha256(source) != expected["sha256"]:
            raise RuntimeError(f"Unexpected {name} Blender template: {source}")
        target = DESTINATION / expected["filename"]
        shutil.copy2(source, target)
        receipt["assets"][name] = {
            **expected,
            "source": str(source.relative_to(ROOT)),
        }

    (DESTINATION / ".controlhair-assets.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared ControlHair Blender templates in {DESTINATION}")


if __name__ == "__main__":
    main()
