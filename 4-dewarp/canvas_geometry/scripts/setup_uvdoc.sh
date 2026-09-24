#!/usr/bin/env bash
set -euo pipefail

target="${1:-third_party/UVDoc}"
if [[ -e "$target" ]]; then
  echo "Target already exists: $target" >&2
  exit 1
fi
git clone --depth 1 https://github.com/tanguymagne/UVDoc.git "$target"
echo "UVDoc and its pretrained model are available in: $target"
