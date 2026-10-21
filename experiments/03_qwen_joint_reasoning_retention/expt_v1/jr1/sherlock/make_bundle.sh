#!/usr/bin/env bash
# Cut the Sherlock upload bundle from a committed tree: code/ = experiments/00_program (frozen utilities the tests
# import), 03 and 04, plus CODE_COMMIT and SUBMIT.md. Usage: make_bundle.sh <out_dir> [commit, default HEAD]
set -euo pipefail
out=$(mkdir -p "$1" && cd "$1" && pwd); rev=${2:-HEAD}
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
commit=$(git -C "$repo" rev-parse "$rev")
[ -z "$(git -C "$repo" status --porcelain -- experiments/03_qwen_joint_reasoning_retention experiments/04_qwen_matched_exposure_schedules)" ] \
  || { echo "uncommitted changes under 03/04: commit first" >&2; exit 1; }
tmp=$(mktemp -d)
git -C "$repo" archive --prefix=code/ "$commit" experiments/00_program experiments/03_qwen_joint_reasoning_retention \
  experiments/04_qwen_matched_exposure_schedules | tar x -C "$tmp"
echo "$commit" > "$tmp/code/CODE_COMMIT"
cp "$tmp/code/experiments/03_qwen_joint_reasoning_retention/expt_v1/jr1/sherlock/SUBMIT.md" "$tmp/code/SUBMIT.md"
# deterministic tarball: fixed order, owner and mtime (the commit time)
mtime=$(git -C "$repo" show -s --format=%cI "$commit")
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="$mtime" -C "$tmp" -cf - code | gzip -n > "$out/jr1_sherlock.tgz"
(cd "$out" && sha256sum jr1_sherlock.tgz > jr1_sherlock.tgz.sha256)
cp "$tmp/code/SUBMIT.md" "$out/SUBMIT.md"
rm -rf "$tmp"
echo "bundle $out/jr1_sherlock.tgz commit $commit sha256 $(cut -d' ' -f1 "$out/jr1_sherlock.tgz.sha256") files $(tar tzf "$out/jr1_sherlock.tgz" | wc -l)"
