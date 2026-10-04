"""Publish only reviewed deterministic aggregate outputs; never model calls."""
import json,re,subprocess
from common import REPO,PROGRAM,RUN_ROOT,HOMES,atomic_json,digest,stamp

ALLOWED={f'experiments/{h}/{p}' for h in HOMES for p in ('results.md','expt_v1/cell_status.json','expt_v1/analysis.json','expt_v1/terminal_tradeoffs.png','expt_v1/language_learning_curves.png')}
ALLOWED.add('experiments/00_program/FINAL_STATUS.md')
SECRET=re.compile(rb'(?:sk-(?:ant-|proj-)[A-Za-z0-9_-]{12,}|AKIA[A-Z0-9]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|(?:/lfs/|/dfs/)|Bearer [A-Za-z0-9._-]{15,})')

def git(*args):
    return subprocess.check_output(['git',*args],cwd=REPO,timeout=120)

def publish():
    from supervisor import validate_gates
    validate_gates()
    if git('diff','--cached','--name-only').strip():raise RuntimeError('unexpected pre-existing staged work; no publication')
    changed=set(git('ls-files','--modified').decode().splitlines())|set(git('ls-files','--others','--exclude-standard').decode().splitlines())
    selected=sorted(changed&ALLOWED)
    if not selected:
        head=git('rev-parse','HEAD').decode().strip()
        if git('ls-remote','origin','refs/heads/main').decode().split()[0]!=head:raise RuntimeError('current commit has not landed on main')
        return None
    for name in selected:
        p=REPO/name
        if p.suffix!='.png' and SECRET.search(p.read_bytes()):raise RuntimeError('private content rejected: '+name)
    git('fetch','origin','main')
    subprocess.run(['git','merge-base','--is-ancestor','origin/main','HEAD'],cwd=REPO,check=True,timeout=30)
    git('add','--',*selected)
    staged=set(git('diff','--cached','--name-only').decode().splitlines())
    if staged!=set(selected):raise RuntimeError('staged scope mismatch')
    git('diff','--cached','--check')
    diff=git('diff','--cached','--binary','--no-ext-diff')
    review=PROGRAM/'runtime/publication';review.mkdir(parents=True,exist_ok=True)
    packet=review/('staged-'+stamp().replace(':','-')+'.diff');packet.write_bytes(diff)
    # Exact staged blobs, not merely a working-tree scan. Generated outputs are
    # allowlisted aggregate schemas; full text diff is retained for inspection.
    for name in selected:
        blob=git('show',':'+name)
        if name.endswith('.json'):json.loads(blob)
        if not name.endswith('.png') and SECRET.search(blob):raise RuntimeError('staged secret check failed')
    git('commit','-m','Report verified unified-training pilot progress')
    head=git('rev-parse','HEAD').decode().strip();git('push','origin','HEAD:main')
    landed=git('ls-remote','origin','refs/heads/main').decode().split()[0]
    if landed!=head:raise RuntimeError('main landing not verified')
    receipt={'timestamp':stamp(),'commit':head,'verified_main':True,'files':selected,'exact_diff_sha256':digest(packet),'checks':'frozen source integrity, generated-output allowlist, exact staged blobs/diff, secret scan, diff whitespace'}
    atomic_json(RUN_ROOT/'publication.json',receipt)
    return receipt

if __name__=='__main__':print(json.dumps(publish()))
