"""Deterministic resource arithmetic; no additional model execution."""
import json,random,unittest
from common import PROGRAM,HOMES,METHODS
from project_schedules import schedule_rates

class CalibrationTests(unittest.TestCase):
    def test_adaptive_vertex_bound_covers_floor_simplex(self):
        record={'objectives':[{'objective':o,'seconds':[t]*6,'charged_tokens':[c]*6} for o,t,c in [('pt',1,10),('sft',2,10),('dpo',2,20),('rl',8,30),('rpt',7,30),('hybrid',9,40),('parallel',10,40)]],'checkpoint_seconds':5,'validation_seconds':3}
        rates,_=schedule_rates(record,256);rng=random.Random(31)
        t=[1+5/256,2+5/256,2+5/256,8+5/256];c=[10,10,20,30]
        for _ in range(1000):
            x=[rng.random() for _ in range(4)];p=[.02+.92*v/sum(x) for v in x]
            actual=(12*sum(t)+8*3+64*sum(a*b for a,b in zip(p,t)))/(12*sum(c)+64*sum(a*b for a,b in zip(p,c)))
            self.assertLessEqual(actual,rates['aioli_objective']+1e-12)
    def test_actual_frozen_prospective_full_denominator_budget(self):
        c=json.loads((PROGRAM/'calibration.json').read_text())
        if not c.get('condition_timeouts'):self.skipTest('final resource selection not yet present')
        total=c['smoke_device_seconds']+c['finalization_reservation_seconds']+52*(c['eval_timeout_seconds']+60)+3*sum(c['train_timeouts'].values())
        self.assertAlmostEqual(total,c['total_forecast_bound_seconds']);self.assertLessEqual(total,48*3600)
        self.assertEqual(len(c['train_timeouts']),16);self.assertGreaterEqual(c['resolved_budget_tokens'],c['minimum_budget_tokens_for_three_controller_sweeps'])
        self.assertEqual(len(HOMES)*len(METHODS)*3,48)

if __name__=='__main__':unittest.main()
