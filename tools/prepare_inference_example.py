#!/usr/bin/env python3
"""Convert an example package into the legacy UniAnimate inference layout."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("example_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_image = args.example_dir / "input.png"
    control_video = args.example_dir / "control.mp4"
    if not input_image.is_file() or not control_video.is_file():
        raise FileNotFoundError("Example package must contain input.png and control.mp4")
    if shutil.which(args.ffmpeg) is None and not Path(args.ffmpeg).is_file():
        raise RuntimeError(f"ffmpeg not found: {args.ffmpeg}")

    condition_dir = args.output_dir / "cond"
    existing = list(condition_dir.glob("*.png")) if condition_dir.exists() else []
    if existing and not args.overwrite:
        raise RuntimeError(
            f"{condition_dir} already contains PNG files; pass --overwrite to replace generated inputs"
        )
    for path in existing:
        path.unlink()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    condition_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_image, args.output_dir / "input_img_padded.png")

    subprocess.run(
        [
            args.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(control_video),
            "-fps_mode",
            "passthrough",
            str(condition_dir / "cond_%04d.png"),
        ],
        check=True,
    )
    frames = sorted(condition_dir.glob("cond_*.png"))
    if not frames:
        raise RuntimeError("ffmpeg produced no condition frames")
    shutil.copy2(frames[0], condition_dir / "pose.png")

    receipt = {
        "source_example": str(args.example_dir.resolve()),
        "input_image": "input_img_padded.png",
        "condition_directory": "cond",
        "condition_frames": len(frames),
        "reference_condition": "cond/pose.png",
    }
    (args.output_dir / "prepared_inputs.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(frames)} frames in {args.output_dir}")


if __name__ == "__main__":
    main()
