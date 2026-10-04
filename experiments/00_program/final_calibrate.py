"""Final per-objective resource bound after review, using development evaluation."""
import fcntl,gc,json,subprocess,time
import torch
from common import PROGRAM,REPO,RUN_ROOT,HOMES,METHODS,atomic_json,digest,event,register_child,stamp,runtime_versions,progress
from calibrate import DEFAULT
from training import Engine
from evaluation import evaluate

SOURCE_NAMES=('common.py','training.py','objectives.py','evaluation.py','calibrate.py','final_calibrate.py','data_manifest.json')

def main():
    register_child(role='final_resource_calibration')
    with (RUN_ROOT/'training-owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4);begin=time.monotonic();old=PROGRAM/'calibration.json'
        if not old.exists():raise RuntimeError('prior calibration not completed; no duplicate preparation')
        original=json.loads(old.read_text())
        if original.get('final_resource_calibration'):raise RuntimeError('final calibration already exists')
        atomic_json(PROGRAM/'runtime/calibration-revised-before-final.json',original)
        source_hashes={str((PROGRAM/n).relative_to(REPO)):digest(PROGRAM/n) for n in SOURCE_NAMES}
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        ready=json.loads((PROGRAM/'readiness.json').read_text());DEFAULT['rl_prefix_fraction']=ready['selected_rl_prefix_fraction']
        records=[];evals=[];bounds={}
        for home in HOMES:
            e=Engine(home,'fixed_joint',202,DEFAULT);e.initialize();rates=[];objective=[]
            for action in ('pt','sft','dpo','rl','rpt','hybrid','parallel'):
                times=[];costs=[];gaps=[]
                for i in range(6):
                    torch.cuda.synchronize();start=time.monotonic();cost=e.work.total
                    d=e.update(action,.7);event(PROGRAM/'runtime/final-calibration-events.jsonl',experiment=home,**d)
                    progress(calibration_status='final per-objective timing',calibration_experiment=home,calibration_objective=action,calibration_update=i)
                    torch.cuda.synchronize();elapsed=time.monotonic()-start
                    times.append(elapsed);costs.append(e.work.total-cost)
                    gap=d.get('sampling_scoring_logp_gap_mean')
                    if gap is not None:gaps.append(gap)
                rate=max(t/c for t,c in zip(times,costs));rates.append(rate)
                objective.append({'objective':action,'seconds':times,'charged_tokens':costs,'maximum_seconds_per_token':rate,'sampling_scoring_logp_gap_mean':gaps})
            torch.cuda.synchronize();start=time.monotonic();e.checkpoint(PROGRAM/'runtime/final-calibration-checkpoint.pt',{'calibration':home});torch.cuda.synchronize()
            checkpoint_seconds=time.monotonic()-start
            start=time.monotonic();cost=e.work.total;e.validate();torch.cuda.synchronize();validation_seconds=time.monotonic()-start
            rates.append(validation_seconds/(e.work.total-cost))
            # A worst-action envelope covers every static phase and every allowed
            # adaptive mixture, including zero-gradient rollout-heavy behavior.
            # Add measured checkpoint cost per the minimum observed update work.
            min_cost=min(min(r['charged_tokens']) for r in objective)
            envelope=max(rates)+checkpoint_seconds/(DEFAULT['checkpoint_every']*min_cost)
            bounds[home]=envelope
            config={**DEFAULT,'development_eval_limit':None}
            result=evaluate(e.model,e.reference,e.tokenizer,'cuda',config,PROGRAM/'runtime/final-calibration-evaluation'/home,{'development_only':True,'experiment':home},smoke=True)
            components={k:result[k]['wall_seconds'] for k in ('wiki','preferences','arc','gsm')}
            sizes={k:result[k]['blocks' if k=='wiki' else 'rows'] for k in components}
            evals.append({'experiment':home,'component_seconds':components,'development_counts':sizes,'work':result['work']})
            record={'experiment':home,'objectives':objective,'checkpoint_seconds':checkpoint_seconds,'validation_seconds':validation_seconds,'minimum_update_tokens':min_cost,'envelope_seconds_per_token':envelope,'peak_memory_bytes':torch.cuda.max_memory_allocated()}
            records.append(record);atomic_json(PROGRAM/'runtime/final-calibration-progress.json',{'rows':records,'evaluations':evals,'timestamp':stamp()})
            print('RESOURCE',home,'envelope',envelope,'checkpoint_seconds',checkpoint_seconds,flush=True)
            del e;gc.collect();torch.cuda.empty_cache()
        data=json.loads((PROGRAM/'data_manifest.json').read_text());names={'wiki':'wiki_test','preferences':'prefs_test','arc':'arc_test','gsm':'gsm_test'}
        evaluation=int(1+2*max(sum(r['component_seconds'][k]*data['splits'][split]['rows']/r['development_counts'][k] for k,split in names.items()) for r in evals))
        spent=original['smoke_device_seconds']+time.monotonic()-begin;finalization=1800
        available=48*3600-spent-finalization-52*(evaluation+60)-48*90
        budget=min(100000000,max(0,int(available/(2*24*sum(bounds.values())))))//100000*100000
        train={home+'/'+method:int(90+2*bounds[home]*budget) for home in HOMES for method in METHODS}
        total=spent+finalization+52*(evaluation+60)+sum(24*max(train[h+'/'+m] for m in METHODS) for h in HOMES)
        # Three complete sweeps plus their intervening production windows must
        # fit even using the largest observed production action work and exact
        # 12-of-each-objective probe counts. This is an engineering lower bound.
        validation_bound=4*DEFAULT['validation_rows']*512+DEFAULT['validation_rows']*(384+DEFAULT['max_new_tokens'])
        sweep={r['experiment']:12*sum(max(x['charged_tokens']) for x in r['objectives'] if x['objective'] in ('pt','sft','dpo','rl'))+8*validation_bound for r in records}
        minimum=max(3*sweep[h]+2*DEFAULT['controller_window']*max(max(x['charged_tokens']) for x in next(r for r in records if r['experiment']==h)['objectives'] if x['objective'] in ('pt','sft','dpo','rl')) for h in HOMES)+5*validation_bound
        result={'final_resource_calibration':True,'timestamp':stamp(),'smoke':True,'rows':records,'evaluations':evals,'source_hashes':source_hashes,'source_commit':source_commit,'source_worktree_dirty':True,'environment':runtime_versions(),'smoke_device_seconds':spent,'resolved_budget_tokens':budget,'eval_timeout_seconds':evaluation,'train_timeouts':train,'finalization_reservation_seconds':finalization,'total_forecast_bound_seconds':total,'fits_48_hours':budget>0 and total<=48*3600,'scientific_minimum_met':budget>=minimum,'minimum_budget_tokens_for_three_controller_sweeps':minimum,'rl_prefix_fraction':DEFAULT['rl_prefix_fraction'],'config':DEFAULT,'limitations':'Worst measured objective rate plus checkpoint work, doubled; full development evaluation extrapolated per dataset and doubled. Hardware changes can still interrupt. Scientific gate is opportunities, not proof of reward robustness.'}
        atomic_json(PROGRAM/'calibration.json',result);print('FINAL_CALIBRATION',json.dumps({k:v for k,v in result.items() if k not in ('rows','evaluations','train_timeouts','source_hashes')}),flush=True)

if __name__=='__main__':main()
