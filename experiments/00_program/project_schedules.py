"""Bound the declared schedules, rather than an impossible all-validation mixture."""
import json
from common import PROGRAM,REPO,HOMES,METHODS,atomic_json,digest,stamp

OBJECTIVES=('pt','sft','dpo','rl')
def ratio(probabilities,times,costs):
    return sum(p*t for p,t in zip(probabilities,times))/sum(p*c for p,c in zip(probabilities,costs))
def schedule_rates(record,checkpoint_every):
    by={x['objective']:(sum(x['seconds'])/len(x['seconds']),sum(x['charged_tokens'])/len(x['charged_tokens'])) for x in record['objectives']}
    t=[by[o][0] for o in OBJECTIVES];c=[by[o][1] for o in OBJECTIVES];cp=record['checkpoint_seconds']/checkpoint_every
    # Add periodic checkpoint time per opportunity; charge no fictitious tokens.
    t=[x+cp for x in t]
    fixed=[.55,.20,.10,.15]
    rates={'sequential':sum(p*ti/ci for p,ti,ci in zip(fixed,t,c)),
           'fixed_joint':ratio(fixed,t,c),
           'smooth_joint':max(ratio(p,t,c) for p in ([.70,.20,.05,.05],[.20,.20,.15,.45]))}
    for name,action in [('chord_objective','hybrid'),('parallel_joint','parallel')]:
        rates[name]=ratio([.55,.10,.35],[by['pt'][0]+cp,by['dpo'][0]+cp,by[action][0]+cp],[by['pt'][1],by['dpo'][1],by[action][1]])
    rates['rpt_inspired']=ratio(fixed,[*t[:3],by['rpt'][0]+cp],[*c[:3],by['rpt'][1]])
    # A full adaptive cycle has exactly twelve of each objective in its 48
    # probes, 64 production draws, eight validations, and .02 probability floors.
    # Linear-fractional extrema over the floor simplex occur at its four vertices.
    # Ignore validation work in the denominator, conservatively charging its time
    # without crediting its tokens. Initial/final partial cycles are bounded below.
    probe_seconds=12*sum(t)+8*record['validation_seconds'];probe_cost=12*sum(c)
    vertices=[]
    for j in range(4):
        p=[.02]*4;p[j]=.94
        vertices.append((probe_seconds+64*sum(pi*ti for pi,ti in zip(p,t)))/(probe_cost+64*sum(pi*ci for pi,ci in zip(p,c))))
    rates['aioli_objective']=rates['validation_progress']=max(vertices)
    return rates,probe_seconds

def main():
    path=PROGRAM/'calibration.json';cal=json.loads(path.read_text())
    if not cal.get('resource_selection') or cal.get('schedule_projection'):raise RuntimeError('need completed resource selection exactly once')
    before=PROGRAM/'runtime/resource-envelope-before-schedule-projection.json';atomic_json(before,cal)
    config=cal['config'];all_rates={};partial={}
    for r in cal['rows']:
        rates,probe_seconds=schedule_rates(r,config['checkpoint_every']);all_rates[r['experiment']]=rates
        # Reserve a complete extra probe sweep plus two checkpoint writes for
        # a partial final cycle and stage boundaries; initial/setup uses 90 s.
        partial[r['experiment']]=2*(probe_seconds+2*r['checkpoint_seconds'])
    spent=cal['smoke_device_seconds'];evaluation=cal['eval_timeout_seconds'];final=cal['finalization_reservation_seconds']
    available=48*3600-spent-final-52*(evaluation+60)-sum(24*(90+partial[h]) for h in HOMES)
    budget=min(100000000,max(0,int(available/(2*24*sum(max(all_rates[h].values()) for h in HOMES)))))//100000*100000
    train={h+'/'+m:int(90+partial[h]+2*all_rates[h][m]*budget) for h in HOMES for m in METHODS}
    total=spent+final+52*(evaluation+60)+sum(24*max(train[h+'/'+m] for m in METHODS) for h in HOMES)
    cal.update(resolved_budget_tokens=budget,train_timeouts=train,total_forecast_bound_seconds=total,fits_48_hours=budget>0 and total<=48*3600,scientific_minimum_met=budget>=cal['minimum_budget_tokens_for_three_controller_sweeps'])
    cal['config']['budget_tokens']=budget
    cal['schedule_projection']={'timestamp':stamp(),'source_packet_sha256':digest(before),'rates_seconds_per_token':all_rates,'partial_cycle_reservations_seconds':partial,'reason':'Respect actual forced 48-probe/64-production cycles. A pure-validation or pure-RL adaptive trajectory is impossible. Use exact floor-simplex rate extrema, 2x timing margin, and extra partial-sweep/checkpoint reservation; unchanged 48-hour and three-sweep requirements.'}
    cal['source_hashes']['experiments/00_program/project_schedules.py']=digest(__file__)
    cal['limitations']+=' Final resource bound integrates declared schedules and exact adaptive probe frequencies, with an extra full-sweep reservation. All forecasts are prospective, based on measured means, not guaranteed hardware service.'
    atomic_json(path,cal);print(json.dumps({k:cal[k] for k in ('resolved_budget_tokens','minimum_budget_tokens_for_three_controller_sweeps','scientific_minimum_met','total_forecast_bound_seconds','fits_48_hours')}))

if __name__=='__main__':main()
