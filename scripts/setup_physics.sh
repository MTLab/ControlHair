#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="python3"
venv_dir="${repo_root}/.venv-physics"
profile="paper"
accept_licenses=0
dry_run=0
download_blender=1
download_checkpoints=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --python) python_bin="$2"; shift 2 ;;
    --venv) venv_dir="$2"; shift 2 ;;
    --profile) profile="$2"; shift 2 ;;
    --accept-restricted-licenses) accept_licenses=1; shift ;;
    --skip-blender) download_blender=0; shift ;;
    --download-checkpoints) download_checkpoints=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ "${accept_licenses}" -ne 1 ]; then
  echo "Review THIRD_PARTY_NOTICES.md and pass --accept-restricted-licenses." >&2
  exit 2
fi

case "${profile}" in
  paper)
    torch_packages=(torch==2.5.0 torchvision==0.20.0)
    torch_index=https://download.pytorch.org/whl/cu124
    ;;
  blackwell)
    torch_packages=(torch==2.12.1 torchvision==0.27.1)
    torch_index=https://download.pytorch.org/whl/cu130
    ;;
  *) echo "Unknown profile: ${profile}" >&2; exit 2 ;;
esac

run() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  if [ "${dry_run}" -eq 0 ]; then "$@"; fi
}

run "${python_bin}" -m venv "${venv_dir}"
venv_python="${venv_dir}/bin/python"
run "${venv_python}" -m pip install --upgrade pip setuptools wheel
run "${venv_python}" -m pip install "${torch_packages[@]}" --index-url "${torch_index}"
run "${venv_python}" -m pip install -r "${repo_root}/requirements-physics.txt"
run bash "${repo_root}/scripts/setup_difflocks.sh"
run "${venv_python}" "${repo_root}/scripts/prepare_external_dependencies.py" \
  --component hairstep --accept-restricted-licenses

template_drive="${repo_root}/physics_simulation/assets/lwk_template_drive.blend"
template_wind="${repo_root}/physics_simulation/assets/lwk_template_wind.blend"
if [ "${dry_run}" -eq 1 ]; then
  run git -C "${repo_root}" lfs pull --include="physics_simulation/assets/*.blend"
elif [ ! -s "${template_drive}" ] || [ ! -s "${template_wind}" ] \
  || [ "$(wc -c < "${template_drive}")" -lt 1000000 ] \
  || [ "$(wc -c < "${template_wind}")" -lt 1000000 ]; then
  if ! git lfs version >/dev/null 2>&1; then
    echo "Git LFS is required to fetch the ControlHair Blender templates." >&2
    exit 2
  fi
  run git -C "${repo_root}" lfs pull --include="physics_simulation/assets/*.blend"
fi
run "${venv_python}" "${repo_root}/physics_simulation/prepare_assets.py"

if [ "${download_checkpoints}" -eq 1 ]; then
  run bash "${repo_root}/third_party/difflocks/download_checkpoints.sh"
  run "${venv_python}" "${repo_root}/scripts/prepare_models.py" \
    --component hairstep_networks --accept-third-party-licenses
fi

blender_dir="${repo_root}/third_party/blender-4.1.1-linux-x64"
blender_bin="${blender_dir}/blender"
if [ "${download_blender}" -eq 1 ]; then
  if [ "${dry_run}" -eq 0 ] && { [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; }; then
    echo "Automatic Blender preparation supports Linux x86_64 only." >&2
    echo "Install Blender 4.1.1 manually and set BLENDER_BIN." >&2
    exit 2
  fi
  if [ ! -x "${blender_bin}" ]; then
    archive="${repo_root}/third_party/blender-4.1.1-linux-x64.tar.xz"
    url="https://download.blender.org/release/Blender4.1/blender-4.1.1-linux-x64.tar.xz"
    expected="ab2ea3fe991601a5e6bd2cda786ecaa919c0b39e0550e59978b5d40270c260d3"
    run curl -fL "${url}" -o "${archive}"
    if [ "${dry_run}" -eq 0 ]; then
      actual="$(shasum -a 256 "${archive}" | awk '{print $1}')"
      [ "${actual}" = "${expected}" ] || { echo "Blender checksum mismatch" >&2; exit 1; }
    fi
    run tar -xJf "${archive}" -C "${repo_root}/third_party"
  fi
  if [ "${dry_run}" -eq 0 ]; then
    "${blender_bin}" --version | head -1 | grep -F "Blender 4.1.1"
  fi
fi

echo "Optional physics environment: ${venv_dir}"
if [ "${download_checkpoints}" -eq 0 ]; then
  echo "Checkpoints were not downloaded. Rerun with --download-checkpoints when needed."
fi
echo "See the optional physics section in README.md."
