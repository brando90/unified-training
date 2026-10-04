"""Masked causal objectives, strict on-policy rollouts, and explicit work charges."""
from __future__ import annotations
import contextlib
import math
import re
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F

@dataclass
class Work:
    counts: dict = field(default_factory=dict)
    def add(self, kind, tokens, multiplier=1):
        assert tokens >= 0 and multiplier > 0
        self.counts[kind] = self.counts.get(kind, 0) + int(tokens)*multiplier
    @property
    def total(self): return sum(self.counts.values())
    def state_dict(self): return self.counts.copy()

def amp(device):
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16) if str(device).startswith('cuda') else contextlib.nullcontext()

def batch_tokens(sequences, starts, pad, device, left=False):
    n = max(map(len, sequences))
    ids=torch.full((len(sequences),n),pad,dtype=torch.long,device=device)
    attention=torch.zeros_like(ids); target=torch.zeros_like(ids,dtype=torch.bool)
    for i,(seq,start) in enumerate(zip(sequences,starts)):
        offset=n-len(seq) if left else 0
        ids[i,offset:offset+len(seq)]=torch.tensor(seq,device=device)
        attention[i,offset:offset+len(seq)]=1
        target[i,offset+start:offset+len(seq)]=True
    target[:,0]=False
    return ids,attention,target

def token_logps(model, sequences, starts, pad, device, work, kind='forward'):
    ids,attn,mask=batch_tokens(sequences,starts,pad,device)
    with amp(device): logits=model(input_ids=ids,attention_mask=attn,use_cache=False).logits[:,:-1]
    # Cross entropy avoids keeping a second B*T*V log-softmax tensor alive.
    logps=-F.cross_entropy(logits.float().transpose(1,2),ids[:,1:],reduction='none')
    work.add(kind,ids.numel())
    return logps,mask[:,1:],ids.numel()

def causal_loss(model, rows, pad, device, work, kind='forward', chord=False):
    sequences=[];starts=[]
    for r in rows:
        if 'ids' in r:sequences.append(r['ids']);starts.append(1)
        else:sequences.append(r['prompt_ids']+r['response_ids']);starts.append(len(r['prompt_ids']))
    lp,mask,tokens=token_logps(model,sequences,starts,pad,device,work,kind)
    weights=mask.to(lp.dtype)
    if chord:
        p=lp.detach().exp(); weights=weights*p*(1-p)
    return -(lp*weights).sum()/mask.sum().clamp_min(1), tokens

def preference_loss(model, reference, rows, pad, device, work, beta=1., kind='forward'):
    sequences=[]; starts=[]
    for key in ('chosen_ids','rejected_ids'):
        for r in rows:sequences.append(r['prompt_ids']+r[key]);starts.append(len(r['prompt_ids']))
    lp,mask,tokens=token_logps(model,sequences,starts,pad,device,work,kind)
    policy=(lp*mask).sum(1)/mask.sum(1)
    with torch.no_grad():
        ref_lp,ref_mask,_=token_logps(reference,sequences,starts,pad,device,work,'reference')
        ref=(ref_lp*ref_mask).sum(1)/ref_mask.sum(1)
    n=len(rows);margin=beta*((policy[:n]-policy[n:])-(ref[:n]-ref[n:]))
    return -F.logsigmoid(margin).mean(),tokens,{
        'implicit_preference': (margin>0).float().tolist(),
        'raw_preference': (policy[:n]>policy[n:]).float().tolist(),
        'margin':margin.detach().float().tolist(),
        'policy_margin':(policy[:n]-policy[n:]).detach().float().tolist()}

def preference_validation_loss(model,rows,pad,device,work):
    sequences=[];starts=[]
    for key in ('chosen_ids','rejected_ids'):
        for r in rows:sequences.append(r['prompt_ids']+r[key]);starts.append(len(r['prompt_ids']))
    lp,mask,_=token_logps(model,sequences,starts,pad,device,work,'validation')
    means=(lp*mask).sum(1)/mask.sum(1);n=len(rows)
    return -F.logsigmoid(means[:n]-means[n:]).mean()

def guided_prompt(row,fraction=0.):
    # Expert prefix is context only. Never include final-answer tokens.
    response=row['response_ids'];n=min(int(len(response)*fraction),row.get('answer_start',max(0,len(response)-8)))
    return (row['prompt_ids']+response[:n])[-384:]

NUMBER=re.compile(r'[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:/\d+)?')
def extract_answer(text):
    # Fixed last numeric answer rule, with GSM's #### delimiter preferred.
    tail=text.rsplit('####',1)[-1] if '####' in text else text
    matches=NUMBER.findall(tail)
    if not matches:return None
    raw=matches[-1].replace(',','')
    try:
        if '/' in raw:
            a,b=raw.split('/'); result=Decimal(a)/Decimal(b)
        else:result=Decimal(raw)
        return str(result.normalize()) if result.is_finite() else None
    except (InvalidOperation,ZeroDivisionError):return None

def math_reward(response, answer):
    found=extract_answer(response); gold=extract_answer(answer)
    return float(found is not None and gold is not None and found==gold)

def prefix_reward(response, continuation_ids, tokenizer):
    if 'Prediction:' not in response:return 0.
    prediction=response.rsplit('Prediction:',1)[-1]
    # One formatting space is ignored; all remaining bytes are judged literally.
    if prediction.startswith(' '):prediction=prediction[1:]
    prediction=prediction.rstrip('\r\n')
    if not prediction:return 0.
    raw=prediction.encode('utf-8'); truth=tokenizer.decode(continuation_ids,skip_special_tokens=True).encode('utf-8')
    boundaries={len(tokenizer.decode(continuation_ids[:i],skip_special_tokens=True).encode('utf-8')) for i in range(1,len(continuation_ids)+1)}
    if truth.startswith(b' '):
        truth=truth[1:];boundaries={b-1 for b in boundaries if b>1}
    return float(truth.startswith(raw) and len(raw) in boundaries)

def loo_advantages(rewards, group_size):
    if group_size<2 or len(rewards)%group_size:raise ValueError('complete independent groups of size >=2 required')
    r=rewards.reshape(-1,group_size)
    return (r-(r.sum(1,keepdim=True)-r)/(group_size-1)).reshape(-1)

@torch.no_grad()
def generate(model, prompts, tokenizer, device, work, max_new, sample, kind='generation',capture_logps=False):
    ids,attention,_=batch_tokens(prompts,[len(p) for p in prompts],tokenizer.eos_token_id,device,left=True)
    def count(module,args,kwargs):
        x=kwargs.get('input_ids',args[0] if args else None)
        if x is not None:work.add(kind,x.numel())
    hook=model.register_forward_pre_hook(count,with_kwargs=True)
    try:
        with amp(device):
            output=model.generate(input_ids=ids,attention_mask=attention,max_new_tokens=max_new,
                                  do_sample=sample,temperature=1. if sample else None,
                                  top_k=0 if sample else None,top_p=1. if sample else None,
                                  return_dict_in_generate=capture_logps,output_scores=capture_logps,
                                  use_cache=True,pad_token_id=tokenizer.eos_token_id,eos_token_id=tokenizer.eos_token_id)
    finally:hook.remove()
    sampling_scores=output.scores if capture_logps else None
    if capture_logps:output=output.sequences
    generated=[];truncated=[]
    for row in output[:,ids.shape[1]:].tolist():
        if tokenizer.eos_token_id in row:
            row=row[:row.index(tokenizer.eos_token_id)+1]; truncated.append(False)
        else:truncated.append(len(row)==max_new)
        generated.append(row)
    if capture_logps:
        sampled=output[:,ids.shape[1]:]
        selected=torch.stack([F.log_softmax(scores.float(),dim=-1).gather(1,sampled[:,t:t+1]).squeeze(1) for t,scores in enumerate(sampling_scores)],dim=1).cpu().tolist()
        sample_lp=[values[:len(row)] for values,row in zip(selected,generated)]
        return generated,truncated,sample_lp
    return generated,truncated

def policy_loss(model, rows, tokenizer, device, work, group_size=2, max_new=128, rpt=False,prefix_fraction=0.):
    prompts=[];targets=[]
    for row in rows:
        if rpt:
            context=row['ids'][:128]; target=row['ids'][128:160]
            text='Predict the continuation of the following text. Reason briefly, then finish with Prediction: followed by the continuation.\nText: '+tokenizer.decode(context)+'\n'
            p=tokenizer.encode(text,add_special_tokens=False)[-256:]
        else:p=guided_prompt(row,prefix_fraction);target=row['answer']
        prompts.extend([p]*group_size);targets.extend([target]*group_size)
    output=generate(model,prompts,tokenizer,device,work,max_new,True,capture_logps=True)
    generated,truncated=output[:2]
    texts=tokenizer.batch_decode(generated,skip_special_tokens=True)
    rewards=torch.tensor([prefix_reward(t,a,tokenizer) if rpt else math_reward(t,a) for t,a in zip(texts,targets)],dtype=torch.float32,device=device)
    adv=loo_advantages(rewards,group_size)
    lp,mask,tokens=token_logps(model,[p+y for p,y in zip(prompts,generated)],list(map(len,prompts)),tokenizer.eos_token_id,device,work)
    # Sequence sums give the score-function gradient of expected sequence reward;
    # length normalization would optimize a different, explicitly biased target.
    loss=-(adv.detach()*(lp*mask).sum(1)).mean()
    gap=[]
    if len(output)==3:
        for i,values in enumerate(output[2]):gap.extend((lp[i][mask[i]].detach().float()-torch.tensor(values,device=device)).abs().tolist())
    return loss,tokens,{'reward_mean':rewards.mean().item(),'reward_sum':rewards.sum().item(),
                       'sampling_scoring_logp_gap_mean':sum(gap)/len(gap) if gap else None,'sampling_scoring_logp_gap_max':max(gap) if gap else None,
                       'rollouts':len(rewards),'zero_variance_groups':int((rewards.reshape(-1,group_size).var(1,unbiased=False)==0).sum()),
                       'groups':len(rows),'truncated':sum(truncated),'zero_gradient':bool((adv==0).all()),
                       'generated_tokens':sum(map(len,generated))}
