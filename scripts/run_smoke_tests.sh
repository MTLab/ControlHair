#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python3 -m unittest discover -s tests -v
python3 tools/verify_examples.py examples

while IFS= read -r script; do
  bash -n "$script"
done < <(find scripts inference physics_simulation control_signal training -type f -name '*.sh' | sort)

echo "ControlHair release smoke tests passed."
