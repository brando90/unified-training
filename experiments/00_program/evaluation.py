"""Full official terminal evaluations with item evidence and separate work ledger."""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from common import atomic_json, event
from objectives import Work, token_logps, preference_loss, generate, math_reward
from training import load_data, rng_state, restore_rng

def batches(rows,n):
    for i in range(0,len(rows),n):yield rows[i:i+n]

def interval(values,seed=191,draws=2000):
    a=np.array(values,dtype=float)
    if not len(a):return {'estimate':None,'interval95':None,'p_value':None}
    if np.isin(a,[0.,1.]).all():
        n=len(a);p=float(a.mean());z=1.959963984540054;den=1+z*z/n
        middle=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        return {'estimate':p,'interval95':[middle-half,middle+half],'p_value':None,'test':'none; Wilson conditional item score interval','n':n}
    rng=np.random.default_rng(seed)
    means=[float(rng.choice(a,len(a),replace=True).mean()) for _ in range(draws)]
    return {'estimate':float(a.mean()),'interval95':[float(x) for x in np.quantile(means,[.025,.975])],
            'p_value':None,'test':'none; conditional item bootstrap','n':len(a)}

@torch.no_grad()
def score_arc_batch(model,rows,tok,device,work):
    sequences=[];starts=[];sizes=[]
    for row in rows:
        p=row['prompt_ids'];sizes.append(len(row['choices']['text']))
        for choice in row['choices']['text']:
            sequences.append(p+tok.encode(' '+choice,add_special_tokens=False));starts.append(len(p))
    if max(map(len,sequences))>model.config.max_position_embeddings:raise ValueError('ARC context overflow; no silent test filtering')
    lp,mask,_=token_logps(model,sequences,starts,tok.eos_token_id,device,work,'eval_forward')
    score=(lp*mask).sum(1);norm=score/mask.sum(1);offset=0;items=[]
    for row,n in zip(rows,sizes):
        raw=score[offset:offset+n];normalized=norm[offset:offset+n];labels=row['choices']['label'];answer=row['answerKey']
        items.append({'id':row['id'],'raw_scores':raw.tolist(),'normalized_scores':normalized.tolist(),'correct_raw':int(labels[int(raw.argmax())]==answer),'correct_normalized':int(labels[int(normalized.argmax())]==answer),'choices':n});offset+=n
    return items

@torch.no_grad()
def evaluate(model,reference,tok,device,config,directory,identity,smoke=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for name in ('wiki','preferences','arc','gsm'):
        with (directory/(name+'.jsonl')).open('x'):pass
    work=Work();start=time.monotonic();saved_rng=rng_state();model.eval();reference.eval()
    # Each dataset's evidence is append-only. A partial evaluation is not retried
    # automatically; prospective recovery can score preserved outputs separately.
    records={};limit=config.get('development_eval_limit',16) if smoke else None
    try:
        rows=load_data('wiki_development' if smoke else 'wiki_test')[:limit];items=[]
        for batch in batches(rows,config['eval_batch_size']):
            lp,mask,_=token_logps(model,[r['ids'] for r in batch],[1]*len(batch),tok.eos_token_id,device,work,'eval_forward')
            for row,total,n in zip(batch,(-(lp*mask).sum(1)).tolist(),mask.sum(1).tolist()):
                rec={'id':row['id'],'nll_sum':total,'tokens':n};items.append(rec)
                event(directory/'wiki.jsonl',**rec)
        total=sum(r['nll_sum'] for r in items);n=sum(r['tokens'] for r in items);mean=total/n
        # Block bootstrap of disjoint 512-token segments, ratio estimator.
        rng=np.random.default_rng(192);sums=np.array([r['nll_sum'] for r in items]);counts=np.array([r['tokens'] for r in items]);boot=[]
        for _ in range(2000):
            ix=rng.integers(0,len(items),len(items));boot.append(float(sums[ix].sum()/counts[ix].sum()))
        bounds=np.quantile(boot,[.025,.975]).tolist()
        records['wiki']={'nll':mean,'nll_interval95':bounds,'perplexity':math.exp(min(mean,700)),
                         'perplexity_interval95':[math.exp(min(x,700)) for x in bounds],'tokens':n,'blocks':len(items),'p_value':None,'resampling_unit':'disjoint corpus block; conditional on model; dependence beyond block unmodeled','wall_seconds':time.monotonic()-start}
        phase_start=time.monotonic()
        rows=load_data('prefs_development' if smoke else 'prefs_test')[:limit];items=[]
        for batch in batches(rows,config['eval_batch_size']):
            loss,_,diag=preference_loss(model,reference,batch,tok.eos_token_id,device,work,kind='eval_forward')
            for i,row in enumerate(batch):
                rec={'id':row['id'],'implicit_correct':diag['implicit_preference'][i],'raw_correct':diag['raw_preference'][i],'margin':diag['margin'][i]};items.append(rec);event(directory/'preferences.jsonl',**rec)
        records['preferences']={'implicit_accuracy':interval([r['implicit_correct'] for r in items]),'raw_accuracy':interval([r['raw_correct'] for r in items]),'ties_are_incorrect':True,'rows':len(items),'wall_seconds':time.monotonic()-phase_start}
        phase_start=time.monotonic()
        rows=load_data('arc_development' if smoke else 'arc_test')[:limit];items=[]
        for group in batches(rows,max(1,config['eval_batch_size']//5)):
            for rec in score_arc_batch(model,group,tok,device,work):
                items.append(rec);event(directory/'arc.jsonl',**rec)
        records['arc']={'accuracy_primary_normalized':interval([r['correct_normalized'] for r in items]),'accuracy_raw':interval([r['correct_raw'] for r in items]),'chance_floor':sum(1/r['choices'] for r in items)/len(items),'rows':len(items),'wall_seconds':time.monotonic()-phase_start}
        phase_start=time.monotonic()
        rows=load_data('gsm_development' if smoke else 'gsm_test')[:limit];items=[]
        for batch in batches(rows,config['eval_batch_size']):
            # All prompts retained. Pythia supports 2048; training's 256 prompt
            # cap does not filter the official test denominator.
            prompts=[r['prompt_ids'] for r in batch]
            if max(map(len,prompts))+config['max_new_tokens']>model.config.max_position_embeddings:raise ValueError('GSM context overflow; no silent test filtering')
            outputs,trunc=generate(model,prompts,tok,device,work,config['max_new_tokens'],False,'eval_generation')
            texts=tok.batch_decode(outputs,skip_special_tokens=True)
            for row,text,t,ids in zip(batch,texts,trunc,outputs):
                rec={'id':row['id'],'response':text,'generated_tokens':len(ids),'truncated':t,'correct':math_reward(text,row['answer'])};items.append(rec);event(directory/'gsm.jsonl',**rec)
        records['gsm']={'exact_match':interval([r['correct'] for r in items]),'rows':len(items),'truncated':sum(r['truncated'] for r in items),'zero_reward_floor':0.,'wall_seconds':time.monotonic()-phase_start}
        records.update(identity=identity,complete=not smoke,smoke=smoke,work=work.state_dict(),forward_equivalent_tokens=work.total,wall_seconds=time.monotonic()-start)
        if not smoke:
            for key,split,count in [('wiki','wiki_test','blocks'),('preferences','prefs_test','rows'),('arc','arc_test','rows'),('gsm','gsm_test','rows')]:
                if records[key][count]!=len(load_data(split)):raise RuntimeError('incomplete evaluation denominator: '+key)
        atomic_json(directory/'summary.json',records)
        return records
    finally:restore_rng(saved_rng)

def paired_items(left,right,key):
    a={r['id']:r[key] for r in left};b={r['id']:r[key] for r in right}
    if a.keys()!=b.keys():return {'status':'missing or mismatched paired items','n_left':len(a),'n_right':len(b)}
    differences=np.array([a[k]-b[k] for k in sorted(a)],dtype=float);rng=np.random.default_rng(191)
    draws=[float(rng.choice(differences,len(differences),replace=True).mean()) for _ in range(2000)]
    return {'estimate':float(differences.mean()),'interval95':np.quantile(draws,[.025,.975]).tolist(),'p_value':None,'test':'none; paired item bootstrap, conditional on models','n':len(differences),'degenerate_sample':bool(differences.std()==0)}
