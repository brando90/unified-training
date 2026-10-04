"""Prospective resource-only selection: batched choices and less frequent checkpoints."""
import fcntl,gc,json,time
import torch
from common import PROGRAM,REPO,RUN_ROOT,HOMES,METHODS,atomic_json,digest,register_child,stamp
from training import make_models,load_data
from evaluation import evaluate

def main():
    register_child(role='resource_resolution')
    with (RUN_ROOT/'training-owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=json.loads((PROGRAM/'calibration.json').read_text())
        if not old.get('final_resource_calibration') or old.get('resource_selection'):raise RuntimeError('need one completed pre-selection timing packet')
        path=PROGRAM/'runtime/resource-measurements-before-selection.json';atomic_json(path,old)
        for name,sha in old['source_hashes'].items():
            if not name.endswith('/evaluation.py') and digest(REPO/name)!=sha:raise RuntimeError('unmeasured training source changed: '+name)
        start=time.monotonic();torch.set_num_threads(4);manifest=json.loads((PROGRAM/'data_manifest.json').read_text());evaluations=[]
        config=old['config'].copy();config['checkpoint_every']=256
        for home in HOMES:
            model,reference,tok=make_models(home,202,'cuda',manifest)
            result=evaluate(model,reference,tok,'cuda',{**config,'development_eval_limit':None},PROGRAM/'runtime/batched-evaluation-calibration'/home,{'development_only':True,'experiment':home,'initialization_only':True},smoke=True)
            evaluations.append({'experiment':home,'component_seconds':{k:result[k]['wall_seconds'] for k in ('wiki','preferences','arc','gsm')},'development_counts':{k:result[k]['blocks' if k=='wiki' else 'rows'] for k in ('wiki','preferences','arc','gsm')},'work':result['work']})
            del model,reference;gc.collect();torch.cuda.empty_cache();print('BATCHED_EVALUATION',home,evaluations[-1],flush=True)
        envelopes={};rate_records=[]
        for r in old['rows']:
            rates={x['objective']:sum(x['seconds'])/sum(x['charged_tokens']) for x in r['objectives']}
            n=config['validation_rows'];pt=load_data('wiki_controller')[:n];sft=load_data('gsm_controller')[:n];prefs=load_data('prefs_controller')[:n]
            minimum_validation_work=n*max(len(x['ids']) for x in pt)+n*max(len(x['prompt_ids'])+len(x['response_ids']) for x in sft)+2*n*max(len(x['prompt_ids'])+len(x[k]) for x in prefs for k in ('chosen_ids','rejected_ids'))
            rates['validation_upper_rate']=r['validation_seconds']/minimum_validation_work
            envelope=max(rates.values())+r['checkpoint_seconds']/(config['checkpoint_every']*r['minimum_update_tokens'])
            envelopes[r['experiment']]=envelope;rate_records.append({'experiment':r['experiment'],'weighted_mean_action_rates':rates,'checkpoint_frequency':config['checkpoint_every'],'seconds_per_token_envelope_before_2x_margin':envelope})
        names={'wiki':'wiki_test','preferences':'prefs_test','arc':'arc_test','gsm':'gsm_test'};forecasts=[]
        for fresh in evaluations:
            previous=next(e for e in old['evaluations'] if e['experiment']==fresh['experiment']);seconds=0
            for key,split in names.items():
                measured=fresh['component_seconds'][key]
                if key!='arc':measured=max(measured,previous['component_seconds'][key])
                seconds+=measured*manifest['splits'][split]['rows']/fresh['development_counts'][key]
            forecasts.append(seconds)
        evaluation=int(1+2*max(forecasts));spent=old['smoke_device_seconds']+time.monotonic()-start;finalization=old['finalization_reservation_seconds']
        available=48*3600-spent-finalization-52*(evaluation+60)-48*90
        budget=min(100000000,max(0,int(available/(2*24*sum(envelopes.values())))))//100000*100000
        train={h+'/'+m:int(90+2*envelopes[h]*budget) for h in HOMES for m in METHODS}
        total=spent+finalization+52*(evaluation+60)+sum(24*max(train[h+'/'+m] for m in METHODS) for h in HOMES)
        config['budget_tokens']=budget
        old.update(resource_selection={'date':stamp(),'reason':'Checkpoint writes consumed 20+ seconds and tiny choice batches wasted device throughput. Select a 256-update checkpoint cadence; retain the same objectives/data/three-sweep minimum. Use slowest measured per-action aggregate rate with a 2x margin; retain all individual times and maxima. No model-performance or test-based choice.','original_measurements_sha256':digest(path),'original_measurement_source_hashes':old['source_hashes'].copy(),'rates':rate_records,'batched_development_evaluations':evaluations},config=config,resolved_budget_tokens=budget,smoke_device_seconds=spent,eval_timeout_seconds=evaluation,train_timeouts=train,total_forecast_bound_seconds=total,fits_48_hours=budget>0 and total<=48*3600,scientific_minimum_met=budget>=old['minimum_budget_tokens_for_three_controller_sweeps'])
        old['source_hashes']={name:digest(REPO/name) for name in old['source_hashes']}
        old['source_hashes']['experiments/00_program/resolve_resources.py']=digest(__file__)
        old['limitations']='Slowest weighted mean objective/validation rate plus measured checkpoint overhead, with 2x margin; every timing sample and earlier forecast retained. Batched evaluation remeasured on full development. This is a prospective forecast, not a guaranteed hardware service rate.'
        atomic_json(PROGRAM/'calibration.json',old)
        print('RESOURCE_SELECTION',json.dumps({k:old[k] for k in ('resolved_budget_tokens','smoke_device_seconds','eval_timeout_seconds','total_forecast_bound_seconds','fits_48_hours','scientific_minimum_met','minimum_budget_tokens_for_three_controller_sweeps')}),flush=True)

if __name__=='__main__':main()
