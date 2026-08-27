#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
example_dir="${1:-${repo_root}/examples/motion_control/motion_01}"
output_dir="${2:-${repo_root}/artifacts/inference/motion_01}"
wan_model_dir="${WAN_MODEL_DIR:-${repo_root}/models/wan/Wan2.1-I2V-14B-720P}"
checkpoint="${CONTROLHAIR_CHECKPOINT:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [ -z "${checkpoint}" ]; then
  echo "Set CONTROLHAIR_CHECKPOINT to a prepared ControlHair checkpoint directory or file." >&2
  echo "See the model preparation section in README.md." >&2
  exit 2
fi

python3 "${repo_root}/tools/prepare_inference_example.py" \
  "${example_dir}" \
  "${output_dir}" \
  --overwrite

unianimate_dir="${repo_root}/third_party/UniAnimate-DiT"
if [ ! -f "${unianimate_dir}/examples/unianimate_wan/inference_unianimate_wan_480p.py" ]; then
  echo "Prepared UniAnimate-DiT checkout not found. Run bash scripts/setup.sh first." >&2
  exit 2
fi

cd "${unianimate_dir}"
python3 examples/unianimate_wan/inference_unianimate_wan_480p.py \
  --lora_path "${checkpoint}" \
  --wan_model_dir "${wan_model_dir}" \
  --working_dir "${output_dir}" \
  --repeat "${CONTROLHAIR_REPEAT:-1}" \
  --num_frames "${CONTROLHAIR_NUM_FRAMES:-81}" \
  --num_inference_steps "${CONTROLHAIR_INFERENCE_STEPS:-50}" \
  --num_persistent_param_in_dit "${CONTROLHAIR_PERSISTENT_DIT_PARAMS:-14000000000}"

echo "Output: ${output_dir}/wan_480P_trial_0.mp4"
