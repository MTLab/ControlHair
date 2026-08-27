#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream_dir="${repo_root}/third_party/difflocks"
upstream_repo="https://github.com/Meshcapade/difflocks.git"
upstream_commit="fcc73747dc60320c30228b6711000a53fc0c9d84"
patch_file="${repo_root}/third_party/patches/difflocks-fcc737-controlhair.patch"
hash_manifest="${repo_root}/third_party/patches/difflocks-fcc737-controlhair.sha256"

mkdir -p "${repo_root}/third_party"
if [ -e "${upstream_dir}" ] && [ ! -d "${upstream_dir}/.git" ]; then
  echo "${upstream_dir} exists but is not a Git checkout." >&2
  exit 1
fi
if [ ! -d "${upstream_dir}/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout \
    "${upstream_repo}" "${upstream_dir}"
fi

git -C "${upstream_dir}" fetch origin "${upstream_commit}"
GIT_LFS_SKIP_SMUDGE=1 git -C "${upstream_dir}" checkout --detach "${upstream_commit}"

if (cd "${upstream_dir}" && shasum -a 256 -c "${hash_manifest}" >/dev/null 2>&1); then
  echo "ControlHair DiffLocks patch is already applied."
elif git -C "${upstream_dir}" apply --whitespace=nowarn --reverse --check "${patch_file}" >/dev/null 2>&1; then
  echo "Detected the already-applied ControlHair DiffLocks patch."
else
  git -C "${upstream_dir}" apply --whitespace=nowarn --check "${patch_file}"
  git -C "${upstream_dir}" apply --whitespace=nowarn "${patch_file}"
fi

actual_commit="$(git -C "${upstream_dir}" rev-parse HEAD)"
if [ "${actual_commit}" != "${upstream_commit}" ]; then
  echo "Expected DiffLocks ${upstream_commit}, got ${actual_commit}." >&2
  exit 1
fi
(cd "${upstream_dir}" && shasum -a 256 -c "${hash_manifest}")

face_asset="${upstream_dir}/inference/assets/face_landmarker.task"
face_sha="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
actual_face_sha="$(shasum -a 256 "${face_asset}" 2>/dev/null | awk '{print $1}' || true)"
if [ "${actual_face_sha}" != "${face_sha}" ]; then
  face_url="https://media.githubusercontent.com/media/Meshcapade/difflocks/${upstream_commit}/inference/assets/face_landmarker.task"
  curl -fL "${face_url}" -o "${face_asset}.download"
  actual_face_sha="$(shasum -a 256 "${face_asset}.download" | awk '{print $1}')"
  [ "${actual_face_sha}" = "${face_sha}" ] || { echo "Face asset checksum mismatch" >&2; exit 1; }
  mv "${face_asset}.download" "${face_asset}"
fi
echo "DiffLocks ready at ${upstream_commit} with the ControlHair patch."
