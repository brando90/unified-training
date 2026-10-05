"""Bounded subscription-CLI owner; detached scientific children survive it.

This is coordination infrastructure, not the scientific training supervisor.
Runtime state and native transcripts belong outside the repository.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


def identity(pid=None):
    pid = pid or os.getpid()
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': fields[19],
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def alive(row):
    try:
        wanted = {k: row[k] for k in ('pid', 'start_ticks', 'boot_id')}
        state = Path(f"/proc/{row['pid']}/stat").read_text().rsplit(')', 1)[1].split()[0]
        return identity(row['pid']) == wanted and state != 'Z'
    except (OSError, KeyError, TypeError):
        return False


def live_children(root):
    return [row for row in read(root / 'children.json', []) if alive(row)
            and row.get('role') not in ('watchdog', 'status_renderer')]


def event(root, status, **fields):
    row = {'run_id': root.name, 'timestamp': stamp(), 'status': status, **fields}
    atomic(root / 'owner_status.json', row)
    with (root / 'owner_events.jsonl').open('a') as out:
        out.write(json.dumps(row) + '\n')
        out.flush()
        os.fsync(out.fileno())
    return row


def heartbeat(root, native=None):
    atomic(root / 'watchdog.json', {
        'run_id': root.name, 'timestamp': stamp(), 'status': 'observing',
        'owner': identity(), 'native': native,
        'native_alive': alive(native) if native else False,
        'children': live_children(root),
        'progress': read(root / 'progress.json', {}),
        'cadence_seconds': 120, 'reboot_recovery': 'not installed',
        'notification': 'owner_events.jsonl and parent repository receipt',
    })


def terminal_complete(root):
    receipt = read(root / 'campaign_completion.json', {})
    return (receipt.get('status') == 'complete'
            and isinstance(receipt.get('expected_cells'), int)
            and receipt['expected_cells'] > 0
            and receipt.get('reconciled_cells') == receipt['expected_cells']
            and receipt.get('artifact_verification_passed') is True
            and bool(receipt.get('published_commit'))
            and not live_children(root))


def prompt(root, repo, manifest, attempt):
    return f'''You are the sole remote implementation and publication owner for {root.name}.
Read and execute experiments/03_qwen_joint_reasoning_retention/expt_v1/cc.md in {repo}.
Read {root / 'private/dispatch.json'} first and acknowledge exact inputs and runtime.
SNAP PRE-APPROVAL (Brando, 2026-09-14): you run with full access (permission prompts and sandbox bypassed). Every command, install, Docker/GPU use, file and git operation (branch, commit, push, PR, merge per Rule 46) and every budgeted experiment call needed for this task is already approved. Never stop to ask for permission or confirmation; treat any approval step as granted, proceed, and log what you did. Pause only for a real blocker (missing credential, exhausted budget, a human-only login step).
Invocation {attempt + 1} of at most {manifest['max_invocations']}. Each is bounded by
{manifest['invocation_seconds']} seconds; cumulative coordinator time cannot exceed
{manifest['coordinator_seconds']} seconds. Preserve completed work and healthy registered
children. This invocation follows a verified stopped previous native writer, if any.
State root is {root}; use the acknowledgement/progress/children schemas in the runbook.
If returning with training_running, register the independent deterministic supervisor and
verify it has started real work. It must finish all admitted cells and publish without needing
the model to stay alive. Keep results.md live and commit meaningful updates.
Write campaign_completion.json only after all prospective cells are reconciled and artifacts
verified, with status=complete, expected_cells, reconciled_cells, artifact_verification_passed,
published_commit. Gated or missing cells remain explicit; completed campaign is not a positive result.
If previous work exists, inspect its checkpoint, native error log and child identities first.
Do not issue a final response merely because a plan exists. Complete implementation and actual
execution. No other model workers, provider keys, paid services, or emails. Read private/coordinator_note.md
at phase boundaries if present. Use deterministic commands for testing, polling and evaluation.
Solve the entire assigned task, including every required file, question and subtask. Produce the
complete required deliverable in the specified output location; an outline, partial draft,
progress report or claim of completion is not a substitute. Save and inspect artifacts, run the
required tests, continue fixing within the declared bounds, preserve incomplete evidence and
report exact failures. Never invent verification or reset budgets.
TLDR: Complete the new joint-training campaign from the saved runbook, with durable execution,
full evidence, tested code and verified publication under the shared 96-device-hour ceiling.
'''


def self_test(root):
    root.mkdir(parents=True, exist_ok=True)
    assert alive(identity())
    assert not alive({**identity(), 'start_ticks': '-1'})
    p = subprocess.Popen(['/bin/sh', '-c', 'exit 7'])
    assert p.wait() == 7
    assert not alive({'pid': p.pid, 'start_ticks': '-1', 'boot_id': 'test'})
    assert not terminal_complete(root)
    receipt = event(root, 'simulated_failure_detected', test=True, returncode=7)
    assert read(root / 'owner_status.json')['status'] == receipt['status']
    atomic(root / 'self_test.json', {'status': 'passed', 'timestamp': stamp(),
                                   'checks': ['identity', 'stale identity', 'failed child',
                                              'incomplete rejected', 'durable event']})


def run(root):
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = read(root / 'private/dispatch.json')
    repo = Path(manifest['repo'])
    state = read(root / 'owner_budget.json', {'attempts': 0, 'elapsed_seconds': 0,
                 'session_id': None, 'started_epoch': time.time()})
    if state.get('active_native'):
        if alive(state['active_native']):
            event(root, 'existing_native_writer_preserved', native=state['active_native'])
            return 2
        state['elapsed_seconds'] += max(0, time.time() - state['active_started_epoch'])
        state.pop('active_native', None)
        state.pop('active_started_epoch', None)
        atomic(root / 'owner_budget.json', state)
    if state['attempts'] == 0:
        for name, expected in manifest['file_hashes'].items():
            if hashlib.sha256((repo / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Input hash mismatch: {name}')
    atomic(root / 'driver_identity.json', {'run_id': root.name, 'timestamp': stamp(),
                                         'process': identity(), 'status': 'owner_running'})
    env = os.environ.copy()
    for key in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GEMINI_API_KEY', 'GOOGLE_API_KEY',
                'ANTHROPIC_AUTH_TOKEN'):
        env.pop(key, None)
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    while time.time() < state['started_epoch'] + manifest['campaign_seconds']:
        if terminal_complete(root):
            event(root, 'complete', receipt=read(root / 'campaign_completion.json'))
            heartbeat(root)
            return 0
        children = live_children(root)
        if children:
            heartbeat(root)
            event(root, 'scientific_children_running', children=children)
            time.sleep(120)
            continue
        if read(root / 'progress.json', {}).get('coordinator_status') == 'blocked':
            event(root, 'blocked', progress=read(root / 'progress.json'))
            return 2
        remaining = manifest['coordinator_seconds'] - state['elapsed_seconds']
        if state['attempts'] >= manifest['max_invocations'] or remaining <= 0:
            event(root, 'coordinator_budget_exhausted', budget=state)
            return 2
        attempt = state['attempts']
        args = [manifest['codex'], 'exec']
        if state['session_id']:
            args += ['resume']
        args += ['--dangerously-bypass-approvals-and-sandbox', '-m', manifest['model'],
                 '-c', f'model_reasoning_effort="{manifest["effort"]}"', '--json',
                 '--output-last-message', str(root / f'last-{attempt}.md')]
        if state['session_id']:
            args += [state['session_id']]
        args += ['-']
        log = root / f'codex-{attempt}.jsonl'
        err = root / f'codex-{attempt}.stderr'
        start = time.time()
        state['attempts'] += 1
        atomic(root / 'owner_budget.json', state)
        with log.open('w') as out, err.open('w') as error:
            proc = subprocess.Popen(args, cwd=repo, env=env, stdin=subprocess.PIPE,
                                    stdout=out, stderr=error, text=True)
            native = identity(proc.pid)
            state['active_native'] = native
            state['active_started_epoch'] = start
            atomic(root / 'owner_budget.json', state)
            event(root, 'executor_running', native=native, model=manifest['model'],
                  effort=manifest['effort'], permissions='danger-full-access/never',
                  attempt=attempt, args=args)
            proc.stdin.write(prompt(root, repo, manifest, attempt))
            proc.stdin.close()
            deadline = start + min(manifest['invocation_seconds'], remaining)
            while proc.poll() is None:
                heartbeat(root, native)
                if time.time() >= deadline:
                    # Never signal its process group or independent scientific children.
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    event(root, 'native_deadline', native=native)
                    break
                time.sleep(min(30, max(0.1, deadline - time.time())))
        assert proc.poll() is not None
        state['elapsed_seconds'] += time.time() - start
        state.pop('active_native', None)
        state.pop('active_started_epoch', None)
        for line in log.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get('type') == 'thread.started':
                state['session_id'] = row.get('thread_id')
        atomic(root / 'owner_budget.json', state)
        event(root, 'executor_exited', returncode=proc.returncode, budget=state)
        tail = (log.read_text()[-18000:] + err.read_text()[-8000:]).lower()
        if any(x in tail for x in ('usage limit reached', 'insufficient_quota',
                                  'refresh_token_reused', 'invalid_grant',
                                  'model is not supported', 'model_not_found')):
            if live_children(root):
                event(root, 'provider_blocked_children_preserved')
                # Continue observing children; never spend repeated access probes.
                state['attempts'] = manifest['max_invocations']
                atomic(root / 'owner_budget.json', state)
            else:
                event(root, 'provider_access_blocked', log=log.name, stderr=err.name)
                return 2
    event(root, 'campaign_deadline', children_preserved=live_children(root))
    return 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--self-test', action='store_true')
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test(arguments.root)
    else:
        raise SystemExit(run(arguments.root))
