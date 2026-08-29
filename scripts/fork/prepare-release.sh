#!/bin/zsh

set -euo pipefail

usage() {
  print -ru2 -- "usage: prepare-release.sh OLLAMA_TAG [MLX_REPOSITORY] [OUTPUT_DIRECTORY]"
  exit 2
}

(( $# >= 1 && $# <= 3 )) || usage
ollama_tag="$1"
script_dir="${0:A:h}"
ollama_repo="${script_dir:h:h}"
mlx_repo="${${2:-${MLX_REPOSITORY:-${ollama_repo:h}/mlx}}:A}"
output_root="${${3:-${RELEASE_WORKTREE_ROOT:-${ollama_repo:h}/release-worktrees}/$ollama_tag}:A}"
ollama_upstream="https://github.com/ollama/ollama.git"
mlx_upstream="https://github.com/ml-explore/mlx.git"

[[ "$ollama_tag" == v<->.<->.<-> ]] || usage
[[ -d "$mlx_repo/.git" ]] || {
  print -ru2 -- "MLX repository is not a Git checkout: $mlx_repo"
  exit 1
}
[[ ! -e "$output_root" ]] || {
  print -ru2 -- "output directory already exists: $output_root"
  exit 1
}

old_ollama_base="$(/usr/bin/jq -r '.ollama.upstream_commit' "$ollama_repo/fork/versions.json")"
old_ollama_patch="$(git -C "$ollama_repo" rev-parse HEAD)"
old_mlx_base="$(/usr/bin/jq -r '.mlx.upstream_commit' "$ollama_repo/fork/versions.json")"
old_mlx_patch="$(/usr/bin/jq -r '.mlx.patched_commit' "$ollama_repo/fork/versions.json")"

git -C "$ollama_repo" fetch "$ollama_upstream" tag "$ollama_tag"
new_ollama_base="$(git -C "$ollama_repo" rev-parse "${ollama_tag}^{commit}")"
new_mlx_base="$(git -C "$ollama_repo" show "$new_ollama_base:MLX_VERSION")"
new_mlx_c_base="$(git -C "$ollama_repo" show "$new_ollama_base:MLX_C_VERSION")"

/bin/mkdir -p "$output_root"
mlx_branch="releases/ollama-${ollama_tag}-global-scale.1"
ollama_branch="releases/${ollama_tag}-fp16.1"
git -C "$mlx_repo" fetch "$mlx_upstream" "$new_mlx_base"
git -C "$mlx_repo" worktree add -b "$mlx_branch" "$output_root/mlx" "$new_mlx_base"
git -C "$output_root/mlx" cherry-pick "$old_mlx_base..$old_mlx_patch"

git -C "$ollama_repo" worktree add -b "$ollama_branch" "$output_root/ollama" "$new_ollama_base"
git -C "$output_root/ollama" cherry-pick "$old_ollama_base..$old_ollama_patch"

print -r -- "Prepared release worktrees:"
print -r -- "  MLX:    $output_root/mlx"
print -r -- "  Ollama: $output_root/ollama"
print -r -- "New pins:"
print -r -- "  Ollama: $new_ollama_base"
print -r -- "  MLX:    $new_mlx_base"
print -r -- "  MLX-C:  $new_mlx_c_base"
print -r -- "Review with:"
print -r -- "  git -C $output_root/mlx range-diff $old_mlx_base..$old_mlx_patch $new_mlx_base..HEAD"
print -r -- "  git -C $output_root/ollama range-diff $old_ollama_base..$old_ollama_patch $new_ollama_base..HEAD"
print -r -- "No branches or tags were pushed."
