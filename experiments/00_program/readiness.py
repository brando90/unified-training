"""Bounded development-only signal check; no test metrics or measured cells."""
import gc,json,math,time
import torch
from common import PROGRAM,HOMES,atomic_json,event,register_child,stamp
from training import Engine,load_data,rng_state,restore_rng
from calibrate import DEFAULT
from objectives import Work,generate,guided_prompt,math_reward,extract_answer,token_logps

def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return {'successes':k,'n':n,'estimate':p,'interval95':[mid-half,mid+half],'method':'Wilson score; development items/groups','p_value':None}

@torch.no_grad()
def math_signal(e,fraction,groups=32):
    rows=load_data('gsm_development')[:groups];w=Work();saved=rng_state();torch.manual_seed(76001)
    rewards=[];zero=0;mixed=0;parser_failures=0;truncated=0
    try:
        for offset in range(0,len(rows),8):
            batch=rows[offset:offset+8];prompts=[guided_prompt(r,fraction) for r in batch for _ in range(4)]
            outputs,trunc=generate(e.model,prompts,e.tokenizer,'cuda',w,e.config['max_new_tokens'],True,'development_generation')
            texts=e.tokenizer.batch_decode(outputs,skip_special_tokens=True)
            for i,r in enumerate(batch):
                values=[math_reward(t,r['answer']) for t in texts[4*i:4*i+4]]
                rewards.extend(values);zero+=int(not any(values));mixed+=int(0<sum(values)<4)
            parser_failures+=sum(extract_answer(t) is None for t in texts);truncated+=sum(trunc)
    finally:restore_rng(saved)
    return {'prefix_fraction':fraction,'correctness':wilson(int(sum(rewards)),len(rewards)),
            'mixed_groups':wilson(mixed,len(rows)),'all_zero_groups':zero,'parser_failures':parser_failures,'truncated':truncated,
            'charged_tokens':w.total,'work':w.state_dict()}

@torch.no_grad()
def arc_signal(e):
    rows=load_data('arc_development')[:128];w=Work();success=0;chance=[]
    for row in rows:
        p=row['prompt_ids'];seq=[p+e.tokenizer.encode(' '+s,add_special_tokens=False) for s in row['choices']['text']]
        lp,mask,_=token_logps(e.model,seq,[len(p)]*len(seq),e.tokenizer.eos_token_id,'cuda',w,'development_forward')
        pred=int(((lp*mask).sum(1)/mask.sum(1)).argmax());success+=row['choices']['label'][pred]==row['answerKey'];chance.append(1/len(seq))
    return {'accuracy':wilson(success,len(rows)),'chance_floor':sum(chance)/len(chance),'charged_tokens':w.total}

def main():
    register_child(role='development_readiness');torch.set_num_threads(4);start=time.monotonic();records=[]
    destination=PROGRAM/'readiness.json'
    if destination.exists():raise RuntimeError('readiness already exists; no result-driven repetition')
    for home in HOMES:
        e=Engine(home,'fixed_joint',101,DEFAULT);e.initialize();base=arc_signal(e);measurements=[]
        for stage in (0,64,128):
            while e.step<stage:
                d=e.update('sft',0.);event(PROGRAM/'runtime/readiness-events.jsonl',experiment=home,kind='development_sft',**d)
            signals=[math_signal(e,f) for f in (0.,.5,.9)]
            measurements.append({'sft_steps':stage,'sft_charged_tokens':e.work.total,'signals':signals})
            atomic_json(PROGRAM/'runtime/readiness-progress.json',{'completed_experiments':records,'current':home,'measurements':measurements,'timestamp':stamp()})
            print('READINESS',home,stage,[(s['prefix_fraction'],s['correctness']['successes'],s['mixed_groups']['successes']) for s in signals],flush=True)
        viable=[s['prefix_fraction'] for m in measurements for s in m['signals'] if s['mixed_groups']['successes']>0]
        records.append({'experiment':home,'initial_arc_development':base,'measurements':measurements,'signal_observed':bool(viable),'minimum_viable_prefix':min(viable) if viable else None})
        del e;gc.collect();torch.cuda.empty_cache()
    # This is an engineering signal gate (one mixed group), not a validated
    # benchmark threshold or evidence that unconditional reasoning is solved.
    viable=all(r['signal_observed'] for r in records)
    chosen=max(r['minimum_viable_prefix'] for r in records) if viable else None
    atomic_json(destination,{'timestamp':stamp(),'development_only':True,'experiments':records,'all_experiments_signal_observed':viable,
                            'selected_rl_prefix_fraction':chosen,'device_seconds':time.monotonic()-start,
                            'admission_threshold_source':'agent-specified engineering feasibility only: at least one mixed-success group among 32 groups of four on development after <=128 SFT steps',
                            'test_evaluation_performed':False,'extra_measured_cells':0})

if __name__=='__main__':main()
