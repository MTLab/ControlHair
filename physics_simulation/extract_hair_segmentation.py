#!/usr/bin/env python3
"""Create a hair mask video by comparing Blender depth passes."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import Imath
import cv2
import numpy as np
import OpenEXR


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def read_depth(path: Path) -> np.ndarray:
    exr = OpenEXR.InputFile(str(path))
    header = exr.header()
    window = header["dataWindow"]
    width = window.max.x - window.min.x + 1
    height = window.max.y - window.min.y + 1
    channels = header["channels"]
    name = next((candidate for candidate in ("Z", "Depth", "V", "R") if candidate in channels), None)
    if name is None:
        raise RuntimeError(f"No depth channel in {path}: {list(channels)}")
    raw = exr.channel(name, Imath.PixelType(Imath.PixelType.FLOAT))
    return np.frombuffer(raw, dtype=np.float32).reshape(height, width)


def extract_masks(with_hair: Path, without_hair: Path, output: Path, fps: int, epsilon: float) -> None:
    hair_files = sorted(with_hair.glob("*.exr"), key=natural_key)
    body_files = sorted(without_hair.glob("*.exr"), key=natural_key)
    if not hair_files or len(hair_files) != len(body_files):
        raise RuntimeError("Blender depth folders must contain matching EXR sequences")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="controlhair-mask-") as temporary:
        temp_dir = Path(temporary)
        for index, (hair_path, body_path) in enumerate(zip(hair_files, body_files)):
            hair_depth = read_depth(hair_path)
            body_depth = read_depth(body_path)
            hair_depth[~np.isfinite(hair_depth)] = 1e10
            body_depth[~np.isfinite(body_depth)] = 1e10
            mask = (hair_depth < body_depth - epsilon).astype(np.uint8) * 255
            cv2.imwrite(str(temp_dir / f"{index:04d}.png"), mask)
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
    parser.add_argument("with_hair", type=Path)
    parser.add_argument("without_hair", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    args = parser.parse_args()
    extract_masks(args.with_hair, args.without_hair, args.output, args.fps, args.epsilon)


if __name__ == "__main__":
    main()
