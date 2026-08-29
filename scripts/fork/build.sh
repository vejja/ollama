#!/bin/zsh

set -euo pipefail

script_dir="${0:A:h}"
repo_root="${script_dir:h:h}"

[[ -f "$repo_root/fork/versions.json" ]] || {
  print -ru2 -- "missing fork/versions.json"
  exit 1
}

expected_mlx="$(/usr/bin/jq -r '.mlx.patched_commit' "$repo_root/fork/versions.json")"
actual_mlx="$(<"$repo_root/MLX_VERSION")"
[[ "$actual_mlx" == "$expected_mlx" ]] || {
  print -ru2 -- "MLX_VERSION does not match fork/versions.json"
  exit 1
}

export VERSION="$(/usr/bin/jq -r '.release' "$repo_root/fork/versions.json")"
cd "$repo_root"
exec ./scripts/build_darwin.sh -a arm64 build
