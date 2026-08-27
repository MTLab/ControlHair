#!/usr/bin/env python3
"""Run DiffLocks hair reconstruction and Blender physics for one portrait."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from extract_hair_segmentation import extract_masks
from extract_rgb import encode_frames
from pad_image import pad_image


ROOT = Path(__file__).resolve().parents[1]
DIFFLOCKS = ROOT / "third_party" / "difflocks"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dynamics", choices=["wind", "motion", "wind-motion", "sudden-wind"], default="wind")
    parser.add_argument("--template", type=Path)
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--stiffness", type=float, default=3.0)
    parser.add_argument("--damping", type=float, default=6.0)
    parser.add_argument("--mass", type=float, default=0.1)
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--camera-angle", type=float, default=0.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--keep-render-passes", action="store_true")
    args = parser.parse_args()

    if not (DIFFLOCKS / ".git").is_dir():
        raise SystemExit("Run scripts/setup_physics.sh first")
    input_image = args.input_image.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    blender = args.blender or Path(
        os.environ.get("BLENDER_BIN", ROOT / "third_party/blender-4.1.1-linux-x64/blender")
    )
    template_name = "lwk_template_drive.blend" if args.dynamics == "motion" else "lwk_template_wind.blend"
    template = args.template or DIFFLOCKS / "inference" / "assets" / template_name
    if not blender.is_file() or not os.access(blender, os.X_OK):
        raise FileNotFoundError(f"Blender executable not found: {blender}")
    if not template.is_file() or template.stat().st_size < 1_000_000:
        raise FileNotFoundError(f"Prepared ControlHair Blender template not found: {template}")

    command = [
        sys.executable, "inference_difflocks.py",
        "--img_path", str(input_image), "--out_path", str(output),
        "--blender_path", str(blender), "--blender_template", str(template),
        "--hair_gen", "--export_alembic", "--do_shrinkwrap",
        "--alembic_resolution", "4", "--blender_strands_subsample", "0.1",
        "--blender_vertex_subsample", "0.1", "--stiffness", str(args.stiffness),
        "--damping", str(args.damping), "--mass", str(args.mass),
        "--gravity", str(args.gravity), "--fixed_cam_deg", str(args.camera_angle),
    ]
    if args.dynamics in {"wind", "wind-motion", "sudden-wind"}:
        command.append("--add_wind")
    if args.dynamics in {"motion", "wind-motion"}:
        command.append("--add_head_motion")
    if args.dynamics == "sudden-wind":
        command.append("--sudden_wind")
    subprocess.run(command, cwd=DIFFLOCKS, check=True)

    with_hair = output / "blender_depth_w_hair"
    without_hair = output / "blender_depth_wo_hair"
    encode_frames(with_hair, output / "sim_rgb.mp4", args.fps)
    encode_frames(without_hair, output / "sim_rgb_wo_hair.mp4", args.fps)
    extract_masks(with_hair, without_hair, output / "hair_segments.mp4", args.fps, 1e-5)
    pad_image(input_image, output / "input.png")
    if not args.keep_render_passes:
        shutil.rmtree(with_hair)
        shutil.rmtree(without_hair)
    print(f"Physics outputs ready in {output}")
    print(f"Next: python -m control_signal.run '{output}'")


if __name__ == "__main__":
    main()
