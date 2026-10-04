"""Allocate predeclared infrastructure time by condition, retaining equal work caps."""
import json
from common import PROGRAM,HOMES,METHODS,atomic_json,digest,stamp

def main():
    path=PROGRAM/'calibration.json';c=json.loads(path.read_text())
    if not c.get('schedule_projection') or c.get('condition_timeouts'):raise RuntimeError('condition resource allocation requires one projected packet')
    before=PROGRAM/'runtime/common-timeout-projection.json';atomic_json(before,c)
    rates=c['schedule_projection']['rates_seconds_per_token'];partial=c['schedule_projection']['partial_cycle_reservations_seconds']
    available=48*3600-c['smoke_device_seconds']-c['finalization_reservation_seconds']-52*(c['eval_timeout_seconds']+60)-sum(24*(90+partial[h]) for h in HOMES)
    budget=min(100000000,max(0,int(available/(2*3*sum(sum(rates[h].values()) for h in HOMES)))))//100000*100000
    timeouts={h+'/'+m:int(90+partial[h]+2*rates[h][m]*budget) for h in HOMES for m in METHODS}
    total=c['smoke_device_seconds']+c['finalization_reservation_seconds']+52*(c['eval_timeout_seconds']+60)+3*sum(timeouts.values())
    c.update(resolved_budget_tokens=budget,train_timeouts=timeouts,total_forecast_bound_seconds=total,fits_48_hours=budget>0 and total<=48*3600,scientific_minimum_met=budget>=c['minimum_budget_tokens_for_three_controller_sweeps'],condition_timeouts=True)
    c['config']['budget_tokens']=budget
    c['condition_timeout_selection']={'timestamp':stamp(),'prior_packet_sha256':digest(before),'reason':'Same charged work cap for every cell; condition-specific predeclared infrastructure timeouts from the same 2x rate margin. Summing actual caps avoids pretending all 48 conditions have the slowest adaptive schedule. No test scores, missing cells or post-freeze changes used.'}
    c['source_hashes']['experiments/00_program/resolve_condition_caps.py']=digest(__file__)
    atomic_json(path,c);print(json.dumps({k:c[k] for k in ('resolved_budget_tokens','minimum_budget_tokens_for_three_controller_sweeps','scientific_minimum_met','total_forecast_bound_seconds','fits_48_hours')}))
if __name__=='__main__':main()
