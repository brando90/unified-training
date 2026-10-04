"""Non-inferential real-data throughput smoke; only validation data is scored."""
from __future__ import annotations
import gc
import json
import time
import torch
from common import PROGRAM,HOMES,METHODS,atomic_json,register_child,event,stamp
from training import Engine
from evaluation import evaluate

DEFAULT={'learning_rate':1e-4,'validation_rows':8,'batch_size':16,'rl_prompts':16,'group_size':4,
         'max_new_tokens':128,'eval_batch_size':32,'budget_tokens':25000000,'rl_prefix_fraction':0.,
         'controller_window':64,'checkpoint_every':32}

def main():
    register_child(role='throughput_smoke')
    ready=json.loads((PROGRAM/'readiness.json').read_text())
    if not ready['all_experiments_signal_observed']:raise RuntimeError('development reward signal gate failed')
    DEFAULT['rl_prefix_fraction']=ready['selected_rl_prefix_fraction']
    torch.set_num_threads(4);start=time.monotonic();rows=[];evaluations=[]
    for experiment in HOMES:
        for method in METHODS:
            torch.cuda.reset_peak_memory_stats();begin=time.monotonic();e=Engine(experiment,method,101,DEFAULT)
            e.initialize();torch.cuda.synchronize();train_start=time.monotonic();cost_start=e.work.total
            # Finish at least one full adaptive sweep. Other methods receive a
            # fixed 48-update sample; measurements never enter final trajectories.
            for i in range(48):
                if method in ('aioli_objective','validation_progress'):d=e.advance()
                else:
                    fraction=i/47;e.refresh_reference(fraction);d=e.update(e.choose(fraction),fraction)
                event(PROGRAM/'runtime/smoke-events.jsonl',experiment=experiment,method=method,**d)
                if e.step%DEFAULT['checkpoint_every']==0:e.checkpoint(PROGRAM/'runtime/smoke-checkpoint.pt',{'smoke':True,'experiment':experiment,'method':method})
            torch.cuda.synchronize();elapsed=time.monotonic()-train_start
            r={'experiment':experiment,'method':method,'steps':e.step,'seconds':elapsed,'charged_tokens':e.work.total-cost_start,
               'seconds_per_charged_token':elapsed/(e.work.total-cost_start),'peak_memory_bytes':torch.cuda.max_memory_allocated(),'controller_iterations':e.controller_iterations,
               'setup_seconds':train_start-begin,'work':e.work.state_dict()}
            rows.append(r);atomic_json(PROGRAM/'runtime/calibration-progress.json',{'rows':rows,'timestamp':stamp()})
            if method=='fixed_joint':
                result=evaluate(e.model,e.reference,e.tokenizer,'cuda',DEFAULT,PROGRAM/'runtime/smoke-evaluation'/experiment,{'smoke':True,'experiment':experiment},smoke=True)
                evaluations.append({'experiment':experiment,'seconds':result['wall_seconds'],'items_each':16,'work':result['work'],'component_seconds':{k:result[k]['wall_seconds'] for k in ('wiki','preferences','arc','gsm')}})
            del e;gc.collect();torch.cuda.empty_cache()
            print('SMOKE',json.dumps(r),flush=True)
    # Full evaluation extrapolation includes all 564 corpus blocks, 256
    # preference pairs, 2376 ARC and 1319 generated GSM items. Multiplying by
    # the sum of denominators/16 is deliberately conservative for mixed work.
    data=json.loads((PROGRAM/'data_manifest.json').read_text())
    names={'wiki':'wiki_test','preferences':'prefs_test','arc':'arc_test','gsm':'gsm_test'}
    eval_per_cell=max(sum(x['component_seconds'][key]*data['splits'][split]['rows']/16 for key,split in names.items()) for x in evaluations)*2
    prior_path=PROGRAM/'runtime/smoke-prior-usage.json'
    prior=json.loads(prior_path.read_text())['device_seconds'] if prior_path.exists() else 0.
    spent=time.monotonic()-start+prior+ready['device_seconds'];finalization=1800
    available=48*3600-spent-finalization-52*(eval_per_cell+1)-52*60
    # Each of the 16 condition rates expands to its three frozen seeds; 2x
    # throughput margin and explicit setup/checkpoint allowance, not equal steps.
    rate_sum=sum(24*max(r['seconds_per_charged_token'] for r in rows if r['experiment']==h) for h in HOMES)
    budget=min(100000000,int(max(0,available-48*90)/(2*rate_sum)))
    budget=int(budget//100000*100000)
    train_bounds={r['experiment']+'/'+r['method']:int(90+2*r['seconds_per_charged_token']*budget) for r in rows}
    total_bound=spent+finalization+52*(eval_per_cell+1)+4*60+sum(24*(60+max(train_bounds[h+'/'+m] for m in METHODS)) for h in HOMES)
    result={'smoke':True,'timestamp':stamp(),'rows':rows,'evaluations':evaluations,'smoke_device_seconds':spent,
            'resolved_budget_tokens':budget,'eval_timeout_seconds':int(eval_per_cell+1),'train_timeouts':train_bounds,
            'finalization_reservation_seconds':finalization,'total_forecast_bound_seconds':total_bound,
            'fits_48_hours':budget>0 and total_bound<=48*3600,'default_budget_tokens':DEFAULT['budget_tokens'],'rl_prefix_fraction':DEFAULT['rl_prefix_fraction'],
            'limitations':'Throughput forecast, not guaranteed runtime; 2x margin; smoke is validation-only; all real smoke work counted separately; bounds remain infrastructure limits.'}
    atomic_json(PROGRAM/'calibration.json',result)
    print('CALIBRATION',json.dumps({k:v for k,v in result.items() if k not in ('rows','evaluations','train_timeouts')}),flush=True)

if __name__=='__main__':main()
