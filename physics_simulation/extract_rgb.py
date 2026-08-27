#!/usr/bin/env python3
"""Encode Blender PNG frames as an H.264 video."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def encode_frames(input_dir: Path, output: Path, fps: int = 15) -> None:
    frames = sorted(input_dir.glob("*.png"), key=natural_key)
    if not frames:
        raise FileNotFoundError(f"No PNG frames in {input_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="controlhair-rgb-") as temporary:
        temp_dir = Path(temporary)
        for index, frame in enumerate(frames):
            target = temp_dir / f"{index:04d}.png"
            try:
                os.link(frame, target)
            except OSError:
                shutil.copy2(frame, target)
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", str(fps), "-i", str(temp_dir / "%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
            ],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    encode_frames(args.input_dir, args.output, args.fps)


if __name__ == "__main__":
    main()
