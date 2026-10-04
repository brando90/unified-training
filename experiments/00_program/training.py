"""Restartable local training engine shared by both frozen experiment manifests."""
from __future__ import annotations
import copy
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from common import DATA, PROGRAM, METHODS, atomic_json, digest, event,fsync_directory
from controller import ObjectiveScheduler, OBJECTIVES, validation_progress
from aioli import AioliController,recover_transfer_matrix
from objectives import Work, causal_loss, preference_loss, preference_validation_loss, policy_loss,generate,math_reward,guided_prompt

ROUTES={'sequential':'sequential','uniform_joint':'uniform','fixed_joint':'fixed',
        'smooth_joint':'smooth','validation_progress':'validation_progress'}

def seed_model(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])

def load_data(split):
    # JSON strings may contain U+0085/U+2028. str.splitlines() incorrectly
    # treats them as record separators; JSONL is delimited by physical LF only.
    path=DATA/(split+'.jsonl');manifest=PROGRAM/'data_manifest.json'
    if manifest.exists() and DATA==PROGRAM/'data/prepared':
        spec=json.loads(manifest.read_text()).get('splits',{}).get(split)
        if spec:path=PROGRAM/spec['file']
    with path.open() as f:return [json.loads(s) for s in f]

class Stream:
    def __init__(self,rows,seed):
        self.rows=rows; self.rng=random.Random(seed);self.order=list(range(len(rows)));self.rng.shuffle(self.order);self.pos=0;self.epochs=0
    def take(self,n):
        out=[]
        for _ in range(n):
            if self.pos==len(self.order):self.rng.shuffle(self.order);self.pos=0;self.epochs+=1
            out.append(self.rows[self.order[self.pos]]);self.pos+=1
        return out
    def state_dict(self):return {'order':self.order.copy(),'pos':self.pos,'epochs':self.epochs,'rng':self.rng.getstate()}
    def load_state_dict(self,s):self.order=s['order'];self.pos=s['pos'];self.epochs=s['epochs'];self.rng.setstate(s['rng'])

def make_models(experiment,seed,device,manifest):
    from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
    key='scratch' if experiment.startswith('01') else 'early'; spec=manifest['models'][key]
    seed_model(seed)
    config=AutoConfig.from_pretrained(spec['id'],revision=spec['revision'],local_files_only=True)
    config.attention_dropout=0.;config.hidden_dropout=0.;config.use_cache=False
    # The source config declares half precision. Keep float32 master weights and
    # Adam moments; autocast only the forward path. Half precision Adam's eps can
    # underflow and produce nonfinite parameters even on the first update.
    if key=='scratch':model=AutoModelForCausalLM.from_config(config,attn_implementation='sdpa',dtype=torch.float32)
    else:model=AutoModelForCausalLM.from_pretrained(spec['id'],revision=spec['revision'],config=config,local_files_only=True,attn_implementation='sdpa',dtype=torch.float32)
    model=model.to(device).eval();reference=copy.deepcopy(model).eval().requires_grad_(False)
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise RuntimeError('float32 master weights required')
    tok=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],local_files_only=True)
    tok.pad_token=tok.eos_token;tok.padding_side='left'
    return model,reference,tok

class Engine:
    def __init__(self,experiment,method,seed,config,device='cuda',models=None):
        if method not in METHODS:raise ValueError(method)
        self.experiment=experiment;self.method=method;self.seed=seed;self.config=config;self.device=device
        manifest=json.loads((PROGRAM/'data_manifest.json').read_text()) if models is None else {}
        self.model,self.reference,self.tokenizer=models or make_models(experiment,seed,device,manifest)
        self.optimizers={k:torch.optim.AdamW(self.model.parameters(),lr=config['learning_rate'],weight_decay=.01) for k in ('pt','sft','dpo','rl','rpt','hybrid')}
        self.optimizer=self.optimizers['pt']
        self.work=Work();self.step=0;self.wall_seconds=0.;self.stage='training';self.next_milestone=0
        self.action_rng=random.Random(seed+10000);self.probe_rng=random.Random(seed+20000)
        self.streams={k:Stream(load_data(v),seed+30000+i) for i,(k,v) in enumerate([('pt','wiki_train'),('sft','gsm_train'),('dpo','prefs_train'),('rl','gsm_train'),('rpt','wiki_train')])}
        self.validation_rows={k:load_data(v)[:config['validation_rows']] for k,v in [('pt','wiki_controller'),('sft','gsm_controller'),('dpo','prefs_controller'),('rl','gsm_controller')]}
        self.initial=None;self.current=None;self.best_pt=None;self.history=[];self.controller_iterations=0
        self.scheduler=AioliController(seed=seed+40000,floor=.02) if method in ('aioli_objective','validation_progress') else ObjectiveScheduler(ROUTES.get(method,'fixed'),seed=seed+40000)
        self.probe_state=None;self.next_probe=0;self.last_metrics={}
        self.reference_refreshes=[]

    def validate(self):
        result={}
        with torch.no_grad():
            for obj,rows in self.validation_rows.items():
                if obj=='dpo':loss=preference_validation_loss(self.model,rows,self.tokenizer.eos_token_id,self.device,self.work)
                elif obj=='rl':
                    state=rng_state();torch.manual_seed(self.seed+60000)
                    try:
                        prompts=[guided_prompt(r,self.config.get('rl_prefix_fraction',0.)) for r in rows]
                        outputs,_=generate(self.model,prompts,self.tokenizer,self.device,self.work,self.config['max_new_tokens'],True,'validation_generation')
                        rewards=[math_reward(t,r['answer']) for t,r in zip(self.tokenizer.batch_decode(outputs,skip_special_tokens=True),rows)]
                        loss=(len(rewards)-sum(rewards)+.5)/(len(rewards)+1)
                    finally:restore_rng(state)
                else:loss,_=causal_loss(self.model,rows,self.tokenizer.eos_token_id,self.device,self.work,kind='validation')
                result[obj]=float(loss)
        if not all(np.isfinite(x) and x>=(0. if self.initial is not None else 1e-30) for x in result.values()):raise ValueError('invalid validation losses')
        self.current=result;self.best_pt=min(self.best_pt or result['pt'],result['pt'])
        return result

    def initialize(self):
        if self.initial is not None:return
        self.initial=self.validate()

    def refresh_reference(self,fraction):
        # Same budget clock for all methods; .75 explicitly captures the staged
        # post-SFT policy before its first preference update.
        due=[x for x in (.1,.2,.3,.4,.5,.6,.7,.75,.8,.9) if fraction>=x and x not in self.reference_refreshes]
        if due:
            self.reference.load_state_dict(self.model.state_dict());self.reference_refreshes.extend(due)

    def parallel_update(self,fraction):
        baseline={k:p.detach().clone() for k,p in self.model.named_parameters()}
        sft=self.update('sft',fraction);after_sft={k:p.detach().clone() for k,p in self.model.named_parameters()}
        with torch.no_grad():
            for k,p in self.model.named_parameters():p.copy_(baseline[k])
        rl=self.update('rl',fraction)
        with torch.no_grad():
            for k,p in self.model.named_parameters():p.copy_(.5*(p+after_sft[k]))
        # Both deltas were computed at the same theta; optimizer moments remain
        # independent. Count one scheduler opportunity, two charged subupdates.
        self.step-=1
        return {'objective':'parallel','step':self.step,'charged':sft['charged']+rl['charged'],'sft':sft,'rl':rl,'gradient_clipped':sft['gradient_clipped'] or rl['gradient_clipped']}

    def update(self,obj,progress):
        if obj=='parallel':return self.parallel_update(progress)
        start=self.work.total; self.optimizer=self.optimizers[obj];self.model.zero_grad(set_to_none=True); diag={}
        batch=self.config['batch_size'];pad=self.tokenizer.eos_token_id
        if obj=='pt':loss,tokens=causal_loss(self.model,self.streams['pt'].take(batch),pad,self.device,self.work)
        elif obj=='sft':loss,tokens=causal_loss(self.model,self.streams['sft'].take(batch),pad,self.device,self.work)
        elif obj=='dpo':loss,tokens,diag=preference_loss(self.model,self.reference,self.streams['dpo'].take(batch),pad,self.device,self.work)
        elif obj in ('rl','rpt','hybrid'):
            rpt=obj=='rpt';rows=self.streams['rpt' if rpt else 'rl'].take(self.config['rl_prompts'])
            loss,tokens,diag=policy_loss(self.model,rows,self.tokenizer,self.device,self.work,self.config['group_size'],self.config['max_new_tokens'],rpt,self.config.get('rl_prefix_fraction',0.))
            if obj=='hybrid':
                sft,n=causal_loss(self.model,self.streams['sft'].take(batch),pad,self.device,self.work,chord=True)
                mu=.9*(1-progress);loss=(1-mu)*loss+mu*sft;tokens+=n;diag['mu']=mu;diag['zero_gradient']=False
        else:raise ValueError(obj)
        if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite {obj} loss')
        if not diag.get('zero_gradient',False):
            loss.backward();self.work.add('backward',tokens,2)
            norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True)
            self.optimizer.step();diag['gradient_norm']=float(norm);diag['gradient_clipped']=float(norm)>1
        else:diag['gradient_norm']=0.;diag['gradient_clipped']=False
        self.step+=1;diag.update(objective=obj,loss=float(loss.detach()),charged=self.work.total-start,step=self.step)
        self.last_metrics=diag
        return diag

    def choose(self,progress):
        if self.method in ('chord_objective','parallel_joint'):
            u=self.action_rng.random();return 'pt' if u<.55 else ('dpo' if u<.65 else ('parallel' if self.method=='parallel_joint' else 'hybrid'))
        obj=self.scheduler.sample(progress)
        return 'rpt' if self.method=='rpt_inspired' and obj=='rl' else obj

    def begin_probe(self):
        if self.method in ('aioli_objective','validation_progress'):
            order=self.scheduler.probe_order(); actions=[]
            # Exactly 9/1/1/1 of 12 equal-size minibatches per smoothed probe.
            for j in order:
                part=[OBJECTIVES[j]]*9+[o for i,o in enumerate(OBJECTIVES) if i!=j]
                self.probe_rng.shuffle(part);actions.append(part)
        else:
            order=list(range(4));self.probe_rng.shuffle(order);actions=[[OBJECTIVES[j]]*3 for j in order]
        self.probe_state={'order':order,'actions':actions,'column':0,'offset':0,'before':None,'start_cost':None,'drops':[[0.]*4 for _ in range(4)],'rewards':{},'costs':{},'action_costs':{o:[] for o in OBJECTIVES}}

    def advance(self):
        self.initialize();fraction=min(1.,self.work.total/self.config['budget_tokens'])
        self.refresh_reference(fraction)
        adaptive=self.method in ('aioli_objective','validation_progress')
        if adaptive and self.probe_state is None and self.step>=self.next_probe:self.begin_probe()
        if self.probe_state is not None:
            p=self.probe_state;c=p['column'];j=p['order'][c]
            if p['before'] is None:
                p['start_cost']=self.work.total;p['before']=self.validate().copy()
            diag=self.update(p['actions'][c][p['offset']],fraction);p['action_costs'][diag['objective']].append(diag['charged']/1e6);p['offset']+=1
            if p['offset']==len(p['actions'][c]):
                after=self.validate();cost=(self.work.total-p['start_cost'])/1e6
                # W describes update-count mixtures. Dividing response columns
                # by unequal trajectory costs before the SAME inverse corrupts
                # recovery and can reverse objective rankings. Charge work in
                # the ledger; Aioli responses use fixed loss scales only.
                for i,o in enumerate(OBJECTIVES):p['drops'][i][j]=(p['before'][o]-after[o])/self.initial[o]
                p['rewards'][OBJECTIVES[j]]=validation_progress(self.initial,p['before'],after);p['costs'][OBJECTIVES[j]]=cost
                p['column']+=1;p['offset']=0;p['before']=None
                if p['column']==4:
                    if self.method=='aioli_objective':self.scheduler.update(p['drops'])
                    else:
                        # Unmix first using count W; only then divide recovered
                        # objective effects by measured per-action cost. Map the
                        # new matrix through W so the existing verified update
                        # path recovers exactly that cost-aware matrix.
                        A=recover_transfer_matrix(p['drops']);costs=[sum(p['action_costs'][o])/len(p['action_costs'][o]) for o in OBJECTIVES]
                        A=[[value/costs[j] for j,value in enumerate(row)] for row in A]
                        W=self.scheduler._W
                        remixed=[[sum(W[j][k]*row[k] for k in range(4)) for j in range(4)] for row in A]
                        self.scheduler.update(remixed);diag['cost_per_action_million_tokens']=dict(zip(OBJECTIVES,costs))
                    self.controller_iterations+=1;self.next_probe=self.step+self.config['controller_window'];self.probe_state=None
                    diag['controller']=self.scheduler.snapshot();diag['controller_iterations']=self.controller_iterations
            return diag
        diag=self.update(self.choose(fraction),fraction)
        if isinstance(self.scheduler,ObjectiveScheduler):self.scheduler.record_cost('rl' if diag['objective'] in ('rpt','hybrid','parallel') else diag['objective'],diag['charged']/1e6)
        return diag

    def checkpoint(self,path,identity):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
        state={'identity':identity,'model':self.model.state_dict(),'reference':self.reference.state_dict(),
               'optimizers':{k:o.state_dict() for k,o in self.optimizers.items()},'rng':rng_state(),'streams':{k:v.state_dict() for k,v in self.streams.items()},
               'scheduler':self.scheduler,'action_rng':self.action_rng.getstate(),'probe_rng':self.probe_rng.getstate(),
               'engine':{k:getattr(self,k) for k in ('step','wall_seconds','stage','next_milestone','initial','current','best_pt','controller_iterations','probe_state','next_probe','last_metrics','reference_refreshes')},'work':self.work.counts}
        torch.save(state,tmp)
        with tmp.open('rb') as f:os.fsync(f.fileno())
        os.replace(tmp,path)
        fsync_directory(path.parent)

    def resume(self,path,identity):
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['identity']!=identity:raise ValueError('checkpoint identity mismatch')
        self.model.load_state_dict(state['model']);self.reference.load_state_dict(state['reference'])
        for name,opt in self.optimizers.items():
            opt.load_state_dict(state['optimizers'][name])
        for k,v in state['streams'].items():self.streams[k].load_state_dict(v)
        self.scheduler=state['scheduler'];self.action_rng.setstate(state['action_rng']);self.probe_rng.setstate(state['probe_rng'])
        for k,v in state['engine'].items():setattr(self,k,v)
        self.work=Work(state['work']);restore_rng(state['rng'])
