#!/usr/bin/env bash
# Cut the Exp 05 upload bundle from a committed tree: code/ = experiments/00_program, 03 (the jr1 package Exp 05
# imports) and 05, plus CODE_COMMIT and SUBMIT.md. Usage: make_bundle.sh <out_dir> [commit, default HEAD]
set -euo pipefail
out=$(mkdir -p "$1" && cd "$1" && pwd); rev=${2:-HEAD}
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
commit=$(git -C "$repo" rev-parse "$rev")
[ -z "$(git -C "$repo" status --porcelain -- experiments/03_qwen_joint_reasoning_retention experiments/05_pythia_scratch_joint_learning)" ] \
  || { echo "uncommitted changes under 03/05: commit first" >&2; exit 1; }
tmp=$(mktemp -d)
git -C "$repo" archive --prefix=code/ "$commit" experiments/00_program experiments/03_qwen_joint_reasoning_retention \
  experiments/05_pythia_scratch_joint_learning | tar x -C "$tmp"
echo "$commit" > "$tmp/code/CODE_COMMIT"
cp "$tmp/code/experiments/05_pythia_scratch_joint_learning/SUBMIT.md" "$tmp/code/SUBMIT.md"
mtime=$(git -C "$repo" show -s --format=%cI "$commit")
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="$mtime" -C "$tmp" -cf - code | gzip -n > "$out/p05_bundle.tgz"
(cd "$out" && sha256sum p05_bundle.tgz > p05_bundle.tgz.sha256)
cp "$tmp/code/SUBMIT.md" "$out/SUBMIT.md"
rm -rf "$tmp"
echo "bundle $out/p05_bundle.tgz commit $commit sha256 $(cut -d' ' -f1 "$out/p05_bundle.tgz.sha256") files $(tar tzf "$out/p05_bundle.tgz" | wc -l)"
