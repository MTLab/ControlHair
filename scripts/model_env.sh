#!/usr/bin/env bash

# Source this file after running scripts/prepare_models.py.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
wan_dir="${WAN_MODEL_DIR:-${repo_root}/models/wan/Wan2.1-I2V-14B-720P}"
unianimate_dir="${UNIANIMATE_MODEL_DIR:-${repo_root}/models/unianimate}"

if [ ! -d "$wan_dir" ]; then
  echo "Missing Wan model directory: $wan_dir" >&2
  return 1 2>/dev/null || exit 1
fi

mapfile -t dit_shards < <(find "$wan_dir" -maxdepth 1 -type f -name 'diffusion_pytorch_model-*.safetensors' | sort)
if [ "${#dit_shards[@]}" -eq 0 ]; then
  echo "No Wan diffusion shards found in $wan_dir" >&2
  return 1 2>/dev/null || exit 1
fi

DIT_PATH="$(IFS=,; echo "${dit_shards[*]}")"
IMAGE_ENCODER_PATH="${wan_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
PRETRAINED_LORA_PATH="${unianimate_dir}/UniAnimate-Wan2.1-14B-Lora-12000.ckpt"
WAN_MODEL_DIR="$wan_dir"
CONTROLHAIR_UNIANIMATE_MODEL_DIR="$unianimate_dir"

export DIT_PATH IMAGE_ENCODER_PATH PRETRAINED_LORA_PATH WAN_MODEL_DIR CONTROLHAIR_UNIANIMATE_MODEL_DIR

echo "Prepared ControlHair model environment:"
echo "  WAN_MODEL_DIR=$WAN_MODEL_DIR"
echo "  IMAGE_ENCODER_PATH=$IMAGE_ENCODER_PATH"
echo "  PRETRAINED_LORA_PATH=$PRETRAINED_LORA_PATH"
echo "  CONTROLHAIR_UNIANIMATE_MODEL_DIR=$CONTROLHAIR_UNIANIMATE_MODEL_DIR"
echo "  DiT shards=${#dit_shards[@]}"
