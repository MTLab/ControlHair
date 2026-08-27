#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${CONTROLHAIR_PYTHON:-python3}"
upstream_dir="${repo_root}/third_party/UniAnimate-DiT"
upstream_repo="https://github.com/ali-vilab/UniAnimate-DiT.git"
upstream_commit="61d882c25385042f0cf5bcdaf6853238d9756d68"
patch_file="${repo_root}/third_party/patches/unianimate-61d882c-controlhair.patch"
hash_manifest="${repo_root}/third_party/patches/unianimate-61d882c-controlhair.sha256"

mkdir -p "${repo_root}/third_party"

if [ -e "${upstream_dir}" ] && [ ! -d "${upstream_dir}/.git" ]; then
  echo "${upstream_dir} exists but is not a Git checkout." >&2
  exit 1
fi

if [ ! -d "${upstream_dir}/.git" ]; then
  git clone --filter=blob:none --no-checkout "${upstream_repo}" "${upstream_dir}"
fi

git -C "${upstream_dir}" fetch origin "${upstream_commit}"
git -C "${upstream_dir}" checkout --detach "${upstream_commit}"

if (cd "${upstream_dir}" && shasum -a 256 -c "${hash_manifest}" >/dev/null 2>&1); then
  echo "ControlHair UniAnimate patch is already applied."
elif git -C "${upstream_dir}" apply --whitespace=nowarn --reverse --check "${patch_file}" >/dev/null 2>&1; then
  echo "Detected the already-applied ControlHair UniAnimate patch."
else
  git -C "${upstream_dir}" apply --whitespace=nowarn --check "${patch_file}"
  git -C "${upstream_dir}" apply --whitespace=nowarn "${patch_file}"
fi

actual_commit="$(git -C "${upstream_dir}" rev-parse HEAD)"
if [ "${actual_commit}" != "${upstream_commit}" ]; then
  echo "Expected UniAnimate-DiT ${upstream_commit}, got ${actual_commit}." >&2
  exit 1
fi
(cd "${upstream_dir}" && shasum -a 256 -c "${hash_manifest}")

"${python_bin}" -m pip install -e "${upstream_dir}"
echo "UniAnimate-DiT ready at ${upstream_commit} with the ControlHair patch."
