#!/usr/bin/env python3
"""Pad a portrait to ControlHair's 480:832 aspect ratio."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def pad_image(source: Path, destination: Path) -> None:
    image = cv2.imread(str(source))
    if image is None:
        raise FileNotFoundError(source)
    height, width = image.shape[:2]
    target_ratio = 480 / 832
    if width / height > target_ratio:
        pad_top, pad_left = round(width / target_ratio) - height, 0
    else:
        pad_top, pad_left = 0, round(height * target_ratio) - width
    color = np.mean(image[max(0, min(10, height - 1)):max(1, min(20, height)), :, :], axis=(0, 1))
    padded = cv2.copyMakeBorder(
        image, pad_top, 0, pad_left, 0, cv2.BORDER_CONSTANT, value=color
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), padded):
        raise RuntimeError(f"Cannot write {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    pad_image(args.source, args.destination)


if __name__ == "__main__":
    main()
