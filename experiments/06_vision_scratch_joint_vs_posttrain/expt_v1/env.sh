# Exp 06 environment. `source env.sh` activates; `bash env.sh build` creates the venv (needs internet).
# CLUSTER=farmshare (default when $SCRATCH is unset; L40S, bf16) or sherlock ($SCRATCH set; V100, fp16, glibc 2.17 pins).
export EXP06_CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${CLUSTER:-}" ]; then
  if [ -n "${SCRATCH:-}" ]; then CLUSTER=sherlock; else CLUSTER=farmshare; fi
fi
export CLUSTER
if [ "$CLUSTER" = sherlock ]; then
  export EXP06_ROOT="${EXP06_ROOT:-$SCRATCH/exp06}"
  export EXP06_VENV="$EXP06_ROOT/venv"
  export EXP06_AMP=fp16                     # V100 = CC 7.0: no bf16
  export GPU_FLAGS="-p gpu -C GPU_GEN:VLT --gres=gpu:1"
  export CPU_FLAGS="-p normal"
  module load python/3.12.1 2>/dev/null || true
  PYBIN=python3
else
  export EXP06_ROOT="${EXP06_ROOT:-$HOME/exp06}"  # FarmShare has no /scratch; home is the 198 TB NFS
  export EXP06_VENV="$HOME/venvs/exp06"
  export EXP06_AMP=bf16                     # L40S (Ada) bf16; fp16+GradScaler stays available as --amp fp16
  export GPU_FLAGS="-p gpu --qos=gpu --gres=gpu:1"
  export CPU_FLAGS="-p normal --exclude=wheat-01"  # wheat-01 lacks python3.12-venv
  PYBIN=/usr/bin/python3.12
fi
export MAIL_FLAGS="--mail-type=END,FAIL --mail-user=brandojazz@gmail.com,brando.science@gmail.com"
mkdir -p "$EXP06_ROOT/logs"

exp06_build() {
  set -e
  "$PYBIN" -m venv "$EXP06_VENV"
  "$EXP06_VENV/bin/pip" install -q --upgrade pip
  if [ "$CLUSTER" = sherlock ]; then
    "$EXP06_VENV/bin/pip" install --only-binary=:all: torch==2.6.0 torchvision==0.21.0 numpy==2.2.6 pillow matplotlib \
      --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
  else
    "$EXP06_VENV/bin/pip" install --only-binary=:all: torch==2.13.0 torchvision numpy pillow matplotlib
  fi
  "$EXP06_VENV/bin/pip" freeze > "$EXP06_ROOT/requirements.lock.txt"
  "$EXP06_VENV/bin/python" -c "import torch, torchvision, numpy; print('torch', torch.__version__, 'tv', torchvision.__version__, 'np', numpy.__version__, 'cuda build', torch.version.cuda)"
}

if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "${1:-}" = build ]; then
  exp06_build
elif [ -x "$EXP06_VENV/bin/python" ]; then
  source "$EXP06_VENV/bin/activate"
fi
