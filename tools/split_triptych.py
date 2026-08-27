#!/usr/bin/env python3
"""Split an equal-width triptych video into a reproducible example package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def executable_exists(executable: str) -> bool:
    return shutil.which(executable) is not None or Path(executable).is_file()


def probe(ffprobe: str, ffmpeg: str, path: Path) -> dict[str, Any]:
    if executable_exists(ffprobe):
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(result.stdout).get("streams", [])
        if not streams:
            raise RuntimeError(f"No video stream found in {path}")
        return streams[0]

    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    video = re.search(r"Video: ([^,]+).*?\b(\d{2,5})x(\d{2,5})\b.*?([\d.]+) fps", result.stderr)
    duration = re.search(r"Duration: (\d+):(\d+):([\d.]+)", result.stderr)
    if not video:
        raise RuntimeError(f"Unable to read video metadata from {path}")
    duration_seconds = None
    if duration:
        duration_seconds = (
            int(duration.group(1)) * 3600
            + int(duration.group(2)) * 60
            + float(duration.group(3))
        )
    fps = float(video.group(4))
    return {
        "codec_name": video.group(1).strip(),
        "width": int(video.group(2)),
        "height": int(video.group(3)),
        "avg_frame_rate": video.group(4),
        "nb_frames": round(duration_seconds * fps) if duration_seconds is not None else None,
        "duration": duration_seconds,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split a left-to-right input/control/output triptych into an example package. "
            "The source width must be divisible by three."
        )
    )
    parser.add_argument("input", type=Path, help="Triptych MP4 to split")
    parser.add_argument("output_dir", type=Path, help="Destination example directory")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument(
        "--license-status",
        default="pending-review",
        choices=["verified", "pending-review"],
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not executable_exists(args.ffmpeg):
        raise RuntimeError(f"Required executable not found: {args.ffmpeg}")

    source_media = probe(args.ffprobe, args.ffmpeg, args.input)
    width = int(source_media["width"])
    height = int(source_media["height"])
    if width % 3:
        raise ValueError(f"Triptych width {width} is not divisible by three")
    panel_width = width // 3

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = args.output_dir / "preview_triptych.mp4"
    input_path = args.output_dir / "input.png"
    control_path = args.output_dir / "control.mp4"
    output_path = args.output_dir / "expected_output.mp4"

    shutil.copy2(args.input, preview_path)

    common = [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(args.input)]
    run(
        common
        + [
            "-vf",
            f"crop={panel_width}:{height}:0:0",
            "-frames:v",
            "1",
            str(input_path),
        ]
    )

    def encode_panel(x_offset: int, destination: Path) -> None:
        run(
            common
            + [
                "-vf",
                f"crop={panel_width}:{height}:{x_offset}:0",
                "-an",
                "-map_metadata",
                "-1",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "slow",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(destination),
            ]
        )

    encode_panel(panel_width, control_path)
    encode_panel(panel_width * 2, output_path)

    metadata = {
        "schema_version": 1,
        "source": {
            "repository": args.source_repository,
            "commit": args.source_commit,
            "path": args.source_path,
            "url": args.source_url,
        },
        "license": {
            "status": args.license_status,
            "note": (
                "Verify the original portrait and video redistribution terms before publication."
            ),
        },
        "layout": {
            "type": "equal_width_triptych",
            "panel_order": ["input", "control", "expected_output"],
            "source_width": width,
            "source_height": height,
            "panel_width": panel_width,
        },
        "source_media": source_media,
        "derivation": {
            "input": "First decoded frame of the left panel; not the original source image.",
            "control": "Middle panel cropped and re-encoded as H.264 CRF 18.",
            "expected_output": "Right panel cropped and re-encoded as H.264 CRF 18.",
        },
        "artifacts": {
            "input": artifact(input_path),
            "control": artifact(control_path),
            "expected_output": artifact(output_path),
            "preview_triptych": artifact(preview_path),
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
