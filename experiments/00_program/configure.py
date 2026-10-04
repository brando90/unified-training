"""Create all 48 prospective cells; freeze only after explicit review and smoke."""
from __future__ import annotations
import argparse
import json
import subprocess
from common import REPO,PROGRAM,HOMES,METHODS,atomic_json,digest
from calibrate import DEFAULT

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--landed-commit');args=p.parse_args()
    cal=json.loads((PROGRAM/'calibration.json').read_text()) if (PROGRAM/'calibration.json').exists() else None
    if (PROGRAM/'frozen_program.json').exists():raise RuntimeError('already frozen; no in-place protocol change allowed')
    if args.freeze:
        a=json.loads((PROGRAM/'acceptance.json').read_text())
        if a['status']!='accepted' or a['plan_status']!='reconciled' or not cal or not cal['fits_48_hours']:raise RuntimeError('review/calibration gate pending')
        if not args.landed_commit:raise RuntimeError('landed reviewed setup required')
        subprocess.run(['git','fetch','origin','main'],cwd=REPO,check=True)
        subprocess.run(['git','merge-base','--is-ancestor',args.landed_commit,'origin/main'],cwd=REPO,check=True)
        from integrity import verify_landed_content,verify_prepared_inputs
        sources=list(PROGRAM.glob('*.py'))+[PROGRAM/n for n in ('data_manifest.json','acceptance.json','calibration.json','readiness.json','IMPLEMENTATION.md','PLAN.md','REVIEW_RECONCILIATION.md')]+[REPO/'experiments'/h/'expt_v1/PROTOCOL.md' for h in HOMES]
        source_hashes={str(p.relative_to(REPO)):digest(p) for p in sources}
        verify_landed_content(args.landed_commit,source_hashes);verify_prepared_inputs()
        for name,sha in cal['source_hashes'].items():
            if digest(REPO/name)!=sha:raise RuntimeError('calibration source changed: '+name)
        if not cal['scientific_minimum_met']:raise RuntimeError('insufficient controller opportunities')
    manifests=[]
    for home in HOMES:
        config=(cal.get('config',DEFAULT) if cal else DEFAULT).copy()
        config['budget_tokens']=cal['resolved_budget_tokens'] if cal else DEFAULT['budget_tokens']
        config['rl_prefix_fraction']=cal['rl_prefix_fraction'] if cal else DEFAULT['rl_prefix_fraction']
        # Shared maximum timeout within an experiment preserves a single config;
        # admission forecast recomputes its full 48-cell bound below.
        config['train_timeout_seconds']=max(cal['train_timeouts'][home+'/'+m] for m in METHODS) if cal else 2400
        config['evaluation_timeout_seconds']=cal['eval_timeout_seconds'] if cal else 900
        config['cell_timeout_seconds']=config['train_timeout_seconds']+config['evaluation_timeout_seconds']+60
        cells=[{'id':f'{method}-seed-{seed}','experiment':home,'method':method,'seed':seed} for method in METHODS for seed in (0,1,2)]
        if cal and cal.get('condition_timeouts'):
            for cell in cells:
                cell['train_timeout_seconds']=cal['train_timeouts'][home+'/'+cell['method']]
                cell['cell_timeout_seconds']=cell['train_timeout_seconds']+config['evaluation_timeout_seconds']+60
        manifests.append({'experiment':home,'status':'frozen' if args.freeze else 'prospective','config':config,'cells':cells})
    if args.freeze:
        bound=sum(c.get('cell_timeout_seconds',m['config']['cell_timeout_seconds']) for m in manifests for c in m['cells'])+4*(cal['eval_timeout_seconds']+60)+cal['smoke_device_seconds']+cal['finalization_reservation_seconds']
        if bound>48*3600:raise RuntimeError(f'actual nested timeout sum {bound} exceeds 48 hours')
    for manifest in manifests:atomic_json(REPO/'experiments'/manifest['experiment']/'expt_v1/manifest.json',manifest)
    if args.freeze:
        files=list(PROGRAM.glob('*.py'))+[PROGRAM/n for n in ('data_manifest.json','acceptance.json','calibration.json','readiness.json','IMPLEMENTATION.md','PLAN.md','REVIEW_RECONCILIATION.md')]
        files += [REPO/'experiments'/h/'expt_v1'/n for h in HOMES for n in ('manifest.json','PROTOCOL.md')]
        hashes={str(p.relative_to(REPO)):digest(p) for p in files}
        atomic_json(PROGRAM/'frozen_program.json',{'hashes':hashes,'landed_source_hashes':source_hashes,'calibration':cal,'environment':cal['environment'],'queue_timeout_seconds':bound-cal['smoke_device_seconds'],'setup_landed_commit':args.landed_commit,'cells':48,'initial_evaluations':4})

if __name__=='__main__':main()
