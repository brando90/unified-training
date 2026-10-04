"""Local-only runtime, atomic receipts, and process fencing for the pilot."""
from __future__ import annotations
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

PROGRAM = Path(__file__).resolve().parent
REPO = PROGRAM.parent.parent
RUN_ROOT = REPO.parent
RUNTIME = PROGRAM / 'runtime'
DATA = PROGRAM / 'data' / 'prepared'
METHODS = ('sequential', 'parallel_joint', 'fixed_joint', 'smooth_joint',
           'aioli_objective', 'validation_progress', 'chord_objective', 'rpt_inspired')
HOMES = ('01_scratch_joint_objectives', '02_early_checkpoint_joint_objectives')

def local_environment():
    for key, name in [('HF_HOME', 'hf_cache'), ('HF_DATASETS_CACHE', 'hf_cache/datasets'),
                      ('XDG_CACHE_HOME', 'runtime/cache'), ('TMPDIR', 'runtime/tmp'),
                      ('TORCH_HOME', 'runtime/torch')]:
        path = PROGRAM / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_XET'] = '1'

def stamp():
    return datetime.now(timezone.utc).isoformat()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_directory(path.parent)

def fsync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def runtime_versions():
    import torch,transformers
    return {'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda,'parameter_dtype':'torch.float32','forward_autocast':'torch.bfloat16','deterministic_gpu_algorithms':False}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def identity(pid=None):
    pid = pid or os.getpid()
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': fields[19],
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}

def alive(record):
    try:
        return identity(record['pid']) == {k: record[k] for k in ('pid', 'start_ticks', 'boot_id')} and Path(f"/proc/{record['pid']}/stat").read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except (OSError, KeyError, ValueError): return False

def register_child(pid=None, role='worker'):
    import fcntl
    with (RUN_ROOT / 'children.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = RUN_ROOT / 'children.json'
        rows = json.loads(path.read_text()) if path.exists() else []
        row = {**identity(pid), 'role': role, 'registered_at': stamp()}
        if not any(x['pid'] == row['pid'] and x['start_ticks'] == row['start_ticks'] for x in rows):
            rows.append(row)
        atomic_json(path, rows)
    return row

def event(path, **record):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps({'timestamp': stamp(), **record}, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())

def progress(**changes):
    import fcntl
    with (RUN_ROOT/'progress.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path = RUN_ROOT / 'progress.json'
        data = json.loads(path.read_text()) if path.exists() else {}
        changed=any(data.get(k)!=v for k,v in changes.items())
        data.update(changes, timestamp=stamp(), run_id=RUN_ROOT.name)
        if changed:data['last_progress_at']=stamp();data['updated_at']=data['last_progress_at']
        atomic_json(path, data)

local_environment()
