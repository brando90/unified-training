"""Analyze all frozen rows; separate training-seed and conditional-item uncertainty."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from common import REPO,HOMES,METHODS,atomic_json
from evaluation import paired_items
from supervisor import update_ledgers

def seed_summary(values):
    if len(values)!=3:return {'status':'incomplete','n':len(values),'expected':3,'values':values}
    a=np.array(values);sd=float(a.std(ddof=1));se=sd/math.sqrt(3);mean=float(a.mean())
    return {'status':'complete','n':3,'values':values,'mean':mean,'sample_sd':sd,
            'interval95':[mean-4.302652729911275*se,mean+4.302652729911275*se],
            'test':'none; Student t interval with 2 degrees of freedom; normal-seed assumption, weak with three seeds','p_value':None}

def read_items(path):
    with Path(path).open() as f:rows=[json.loads(l) for l in f]
    for r in rows:
        if 'nll_sum' in r:r['nll']=r['nll_sum']/r['tokens']
    return rows

def two_stage(arrays):
    if len(arrays)!=3:return {'status':'incomplete','available_seeds':len(arrays),'expected_seeds':3}
    a=np.asarray(arrays,dtype=float);rng=np.random.default_rng(901);draws=[]
    for _ in range(2000):
        seeds=rng.integers(0,3,3)
        # Same item resample across selected seeds preserves shared-item
        # dependence; within each seed paired method differences stay paired.
        items=rng.integers(0,a.shape[1],a.shape[1]);draws.append(float(a[seeds][:,items].mean()))
    return {'estimate':float(a.mean()),'interval95':np.quantile(draws,[.025,.975]).tolist(),'p_value':None,'test':'none; two-stage paired seed/item bootstrap; only three seeds, descriptive','n_seeds':3,'n_items':a.shape[1]}

def missing_accuracy_bounds(values):
    return {'observed_seeds':len(values),'expected_seeds':3,'complete_case_mean':float(np.mean(values)) if values else None,'all_seed_possible_mean':[sum(values)/3,(sum(values)+3-len(values))/3],'interpretation':'identification bounds, not a confidence interval; missing accuracy in [0,1]'}

def main():
    from supervisor import validate_gates,completion_valid
    from integrity import verify_cell_artifacts
    validate_gates()
    manifests=[json.loads((REPO/'experiments'/h/'expt_v1/manifest.json').read_text()) for h in HOMES]
    update_ledgers(manifests)
    for m in manifests:
        home=REPO/'experiments'/m['experiment'];complete={};aggregate={};pairs=[]
        for c in m['cells']:
            root=home/'expt_v1/runtime'/c['id'];p=root/'receipt.json'
            if p.exists() and json.loads(p.read_text()).get('status')=='complete':
                receipt=json.loads(p.read_text())
                if not completion_valid(0,receipt) or not verify_cell_artifacts(root,c):raise RuntimeError('unverified completed cell: '+c['id'])
                complete[c['method'],c['seed']]={'root':root,'eval':json.loads((root/'evaluation/summary.json').read_text()),'train':json.loads((root/'training.json').read_text())}
        for method in METHODS:
            group=[complete[method,s] for s in range(3) if (method,s) in complete]
            aggregate[method]={
                'arc':seed_summary([r['eval']['arc']['accuracy_primary_normalized']['estimate'] for r in group]),
                'gsm':seed_summary([r['eval']['gsm']['exact_match']['estimate'] for r in group]),
                'wiki_nll':seed_summary([r['eval']['wiki']['nll'] for r in group]),
                'preference_raw_accuracy':seed_summary([r['eval']['preferences']['raw_accuracy']['estimate'] for r in group]),
                'preference_implicit_accuracy':seed_summary([r['eval']['preferences']['implicit_accuracy']['estimate'] for r in group]),
                'missing_arc_sensitivity':missing_accuracy_bounds([r['eval']['arc']['accuracy_primary_normalized']['estimate'] for r in group]),
                'missing_gsm_sensitivity':missing_accuracy_bounds([r['eval']['gsm']['exact_match']['estimate'] for r in group]),
                'training_seconds':sum(r['train']['wall_seconds'] for r in group),
                'evaluation_seconds':sum(r['eval']['wall_seconds'] for r in group)}
            for benchmark,file,key in [('arc','arc.jsonl','correct_normalized'),('gsm','gsm.jsonl','correct'),('preference','preferences.jsonl','raw_correct'),('preference_implicit','preferences.jsonl','implicit_correct'),('wiki','wiki.jsonl','nll')]:
                arrays=[[v[key] for v in sorted(read_items(r['root']/'evaluation'/file),key=lambda v:v['id'])] for r in group]
                aggregate[method][benchmark+'_two_stage']=two_stage(arrays)
            aggregate[method]['controller_iterations']=[r['train']['controller_iterations'] for r in group]
            aggregate[method]['realized_work_by_seed']=[r['train']['work'] for r in group]
            if method=='validation_progress':continue
            for benchmark,file,key in [('arc','arc.jsonl','correct_normalized'),('gsm','gsm.jsonl','correct'),('preference','preferences.jsonl','raw_correct'),('preference_implicit','preferences.jsonl','implicit_correct'),('wiki','wiki.jsonl','nll')]:
                diffs=[];arrays=[]
                for seed in range(3):
                    if ('validation_progress',seed) not in complete or (method,seed) not in complete:continue
                    a=complete['validation_progress',seed];b=complete[method,seed]
                    paired=paired_items(read_items(a['root']/'evaluation'/file),read_items(b['root']/'evaluation'/file),key)
                    if 'estimate' not in paired:raise RuntimeError('paired item identity mismatch')
                    pairs.append({'method_minus_baseline':'validation_progress minus '+method,'benchmark':benchmark,'seed':seed,'conditional_item_difference':paired})
                    diffs.append(paired['estimate'])
                    av={x['id']:x[key] for x in read_items(a['root']/'evaluation'/file)};bv={x['id']:x[key] for x in read_items(b['root']/'evaluation'/file)}
                    arrays.append([av[k]-bv[k] for k in sorted(av)])
                primary='wiki' if m['experiment'].startswith('01') else 'arc'
                pairs.append({'method_minus_baseline':'validation_progress minus '+method,'benchmark':benchmark,'paired_seed_difference':seed_summary(diffs),'two_stage_difference':two_stage(arrays),'primary_contrast':method in ('sequential','aioli_objective') and benchmark==primary})
        initial_changes=[]
        for (method,seed),r in complete.items():
            initial=home/'expt_v1/runtime'/f'initial-seed-{seed if m["experiment"].startswith("01") else 0}'
            if (initial/'summary.json').exists():
                initial_changes.append({'method':method,'seed':seed,'untouched_summary':json.loads((initial/'summary.json').read_text()),'arc_change_from_initial':paired_items(read_items(r['root']/'evaluation/arc.jsonl'),read_items(initial/'arc.jsonl'),'correct_normalized'),'wiki_change_from_initial':paired_items(read_items(r['root']/'evaluation/wiki.jsonl'),read_items(initial/'wiki.jsonl'),'nll')})
        result={'complete':len(complete),'expected':24,'seed_summary':aggregate,'paired_comparisons':pairs,'initial_changes':initial_changes,'hypothesis_tests':'none; exploratory pilot, no universal-win inference; missing-value identification bounds separate from complete-case uncertainty'}
        atomic_json(home/'expt_v1/analysis.json',result)
        # Standalone figures use standard plotting and contain no private paths.
        if complete:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(1,3,figsize=(14,4))
            for method in METHODS:
                group=[complete[method,s] for s in range(3) if (method,s) in complete]
                x=[r['train']['charged_tokens']/1e6 for r in group]
                for axis,ys in zip(ax,[[r['eval']['wiki']['nll'] for r in group],[r['eval']['arc']['accuracy_primary_normalized']['estimate'] for r in group],[r['eval']['gsm']['exact_match']['estimate'] for r in group]]):axis.scatter(x,ys,label=method,s=20)
            for axis,label in zip(ax,['WikiText negative log likelihood','ARC-Easy accuracy','GSM8K exact match']):axis.set_xlabel('Training work (million forward-equivalent tokens)');axis.set_ylabel(label)
            ax[0].legend(fontsize=6);fig.tight_layout();fig.savefig(home/'expt_v1/terminal_tradeoffs.png',dpi=160);plt.close(fig)
            fig,axis=plt.subplots(figsize=(8,4))
            for (method,seed),r in complete.items():
                events=read_items(r['root']/'events.jsonl');curve=[e for e in events if e['kind']=='validation_milestone']
                axis.plot([e['charged_total']/1e6 for e in curve],[e['losses']['pt'] for e in curve],alpha=.6,label=f'{method}/{seed}')
            axis.set_xlabel('Inclusive training work (million forward-equivalent tokens)');axis.set_ylabel('Validation text negative log likelihood');axis.legend(fontsize=5,ncol=3);fig.tight_layout();fig.savefig(home/'expt_v1/language_learning_curves.png',dpi=160);plt.close(fig)
        with (home/'results.md').open('a') as f:
            f.write('\nSeed uncertainty, two-stage seed/item intervals, initial-checkpoint changes and all paired comparisons: `expt_v1/analysis.json`. Missing trained seeds remain missing with separate accuracy sensitivity bounds. p-val=n/a throughout; no formal superiority verdict.\n')
            if complete:
                f.write('\n![Terminal tradeoffs](expt_v1/terminal_tradeoffs.png)\n\n**Each point represents one completed seed.** Endpoints are shown against inclusive training work; missing cells remain in the table and no frontier superiority is claimed.\n\n![Language learning curves](expt_v1/language_learning_curves.png)\n\n**Controller-set language loss is observed at fixed work milestones.** Curves are descriptive monitoring evidence, separate from development selection and terminal test results.\n')

if __name__=='__main__':main()
