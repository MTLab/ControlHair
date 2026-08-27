#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="python3"
venv_dir="${repo_root}/.venv"
dry_run=0
profile="paper"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --python)
      python_bin="$2"
      shift 2
      ;;
    --venv)
      venv_dir="$2"
      shift 2
      ;;
    --profile)
      profile="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$profile" in
  paper)
    torch_packages=(torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0)
    torch_index=https://download.pytorch.org/whl/cu124
    ;;
  blackwell)
    torch_packages=(torch==2.12.1 torchvision==0.27.1)
    torch_index=https://download.pytorch.org/whl/cu130
    ;;
  *)
    echo "Unknown profile: $profile (expected paper or blackwell)" >&2
    exit 2
    ;;
esac

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [ "$dry_run" -eq 0 ]; then
    "$@"
  fi
}

if [ "$dry_run" -eq 0 ] && ! command -v ffmpeg >/dev/null 2>&1; then
  echo "FFmpeg is required for example and diffusion inference." >&2
  echo "Install it with your operating system package manager, then rerun setup." >&2
  exit 2
fi

run "$python_bin" -m venv "$venv_dir"
venv_python="${venv_dir}/bin/python"
run "$venv_python" -m pip install --upgrade pip setuptools wheel
run "$venv_python" -m pip install "${torch_packages[@]}" --index-url "$torch_index"
run "$venv_python" -m pip install -r "${repo_root}/requirements-release.txt"
run env CONTROLHAIR_PYTHON="$venv_python" bash "${repo_root}/scripts/setup_unianimate.sh"

echo "Environment path: ${venv_dir}"
echo "Environment profile: ${profile}"
echo "Prepare model assets with scripts/prepare_models.py."
