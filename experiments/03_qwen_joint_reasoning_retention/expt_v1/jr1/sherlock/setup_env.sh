#!/usr/bin/env bash
# Build the JR1 venv on $SCRATCH from the hashed lock (wheel-only: Sherlock is glibc 2.17, source builds fail).
# Idempotent: a venv with .ready and the same lock hash is kept. Run on the login node or inside prep.sbatch.
set -euo pipefail
source "${JR1_SH:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/env.sh"
LOCK=$JR1_SH/requirements-sherlock.lock
WANT=$(sha256sum "$LOCK" | cut -d' ' -f1)
if [ -f "$JR1_VENV/.ready" ] && [ "$(cat "$JR1_VENV/.ready")" = "$WANT" ]; then
  echo "venv ready: $JR1_VENV (lock $WANT)"; exit 0
fi
PYBIN=${JR1_BASE_PYTHON:-python3.12}
command -v "$PYBIN" >/dev/null || PYBIN=python3
"$PYBIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' 
rm -rf "$JR1_VENV"
"$PYBIN" -m venv "$JR1_VENV"
"$JR1_VENV/bin/pip" install -q --upgrade "pip==25.2"
"$JR1_VENV/bin/pip" install --only-binary=:all: --require-hashes --no-deps -r "$LOCK"
"$JR1_VENV/bin/pip" check
"$JR1_VENV/bin/python" - <<'PY'
import torch, transformers, pyarrow, math_verify, huggingface_hub, sys
print({"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
       "transformers": transformers.__version__, "pyarrow": pyarrow.__version__, "hub": huggingface_hub.__version__})
PY
echo "$WANT" > "$JR1_VENV/.ready"
echo "venv built: $JR1_VENV"
