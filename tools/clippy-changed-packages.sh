#!/usr/bin/env bash
#
# Runs `cargo clippy` scoped to the workspace packages that own the changed files
# passed as arguments (deduped). Intended as a prek/pre-commit hook entry with
# `pass_filenames = true` and `types = ["rust"]`.
#
# Exits 0 (no-op) if no argument resolves to a workspace package.
set -euo pipefail

(($# == 0)) && exit 0

# "<manifest-dir>\t<package-name>" for every workspace member (no registry deps).
members="$(cargo metadata --no-deps --format-version 1 \
	| jq -r '.packages[] | "\(.manifest_path | rtrimstr("/Cargo.toml"))\t\(.name)"')"

root="$(pwd)"
declare -A seen=()
args=()
for f in "$@"; do
	abs="$root/$f"
	best_dir=""
	best_name=""
	# Pick the package whose manifest directory is the longest prefix of the file,
	# so nested members (e.g. crates/frontend/ceck) resolve correctly.
	while IFS=$'\t' read -r dir name; do
		case "$abs" in
		"$dir"/*) (("${#dir}" > "${#best_dir}")) && {
			best_dir="$dir"
			best_name="$name"
		} ;;
		esac
	done <<<"$members" # here-string, not a pipe, so state survives in this shell
	if [[ -n $best_name && -z ${seen["$best_name"]:-} ]]; then
		seen["$best_name"]=1
		args+=(--package "$best_name")
	fi
done

((${#args[@]} == 0)) && exit 0

echo "clippy: cargo clippy ${args[*]} --all-targets" >&2
exec cargo clippy "${args[@]}" --all-targets -- -D warnings
