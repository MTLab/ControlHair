#!/usr/bin/env python3
"""Convert physics outputs into a ControlHair conditioning video."""

from __future__ import annotations

import argparse
from pathlib import Path

from .extract import WrappedVideoPreprocessorAlign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("simulation_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--disable-align", action="store_true")
    args = parser.parse_args()
    root = args.simulation_dir.expanduser().resolve()
    required = {
        "simulation RGB": root / "sim_rgb.mp4",
        "hairless RGB": root / "sim_rgb_wo_hair.mp4",
        "hair mask": root / "hair_segments.mp4",
        "reference image": root / "input.png",
    }
    for label, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    output = (args.output or root / "control.mp4").expanduser().resolve()
    processor = WrappedVideoPreprocessorAlign(
        drop_face=True,
        disable_align=args.disable_align,
    )
    processor(
        str(required["simulation RGB"]),
        str(required["reference image"]),
        str(output),
        str(required["hair mask"]),
        str(required["hairless RGB"]),
    )
    print(f"ControlHair example package ready: {root}")
    print(f"Run: bash inference/run_example.sh '{root}'")


if __name__ == "__main__":
    main()
