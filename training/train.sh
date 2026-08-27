#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream_dir="${repo_root}/third_party/UniAnimate-DiT"
python_bin="${CONTROLHAIR_PYTHON:-python3}"

if [ ! -f "${upstream_dir}/examples/unianimate_wan/train_unianimate_wan.py" ]; then
  echo "Prepared UniAnimate-DiT checkout not found." >&2
  echo "Run bash scripts/setup.sh first." >&2
  exit 2
fi

: "${DATASET_PATH:?Set DATASET_PATH to the prepared hair dataset}"
: "${DIT_PATH:?Set DIT_PATH to the comma-separated Wan diffusion shard paths}"
: "${IMAGE_ENCODER_PATH:?Set IMAGE_ENCODER_PATH to the Wan image encoder checkpoint}"
: "${PRETRAINED_LORA_PATH:?Set PRETRAINED_LORA_PATH to the UniAnimate LoRA checkpoint}"

wan_model_dir="${WAN_MODEL_DIR:-${repo_root}/models/wan/Wan2.1-I2V-14B-720P}"
text_encoder_path="${TEXT_ENCODER_PATH:-${wan_model_dir}/models_t5_umt5-xxl-enc-bf16.pth}"
vae_path="${VAE_PATH:-${wan_model_dir}/Wan2.1_VAE.pth}"
output_path="${OUTPUT_PATH:-${repo_root}/artifacts/training/controlhair}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "${output_path}"

cd "${upstream_dir}"
"${python_bin}" examples/unianimate_wan/train_unianimate_wan.py \
  --task train \
  --train_architecture lora \
  --lora_rank "${LORA_RANK:-128}" \
  --lora_alpha "${LORA_ALPHA:-128}" \
  --dataset_path "${DATASET_PATH}" \
  --output_path "${output_path}" \
  --dit_path "${DIT_PATH}" \
  --text_encoder_path "${text_encoder_path}" \
  --vae_path "${vae_path}" \
  --image_encoder_path "${IMAGE_ENCODER_PATH}" \
  --pretrained_lora_path "${PRETRAINED_LORA_PATH}" \
  --max_steps "${MAX_STEPS:-10000}" \
  --max_epochs "${MAX_EPOCHS:-5}" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --accumulate_grad_batches "${ACCUMULATE_GRAD_BATCHES:-1}" \
  --training_strategy "${TRAINING_STRATEGY:-deepspeed_stage_2}" \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload
