"""Real process recovery drill with artificial receipts, never experiment cells."""
import json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import common,supervisor
from common import atomic_json,identity,stamp

class RecoveryTests(unittest.TestCase):
    def test_recover_dead_cell_then_execute_pending_cell_and_initials(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);repo=root/'repo';run=root/'run';program=repo/'experiments/00_program'
            run.mkdir();(program/'runtime').mkdir(parents=True);atomic_json(run/'children.json',[])
            atomic_json(program/'runtime/supervisor.json',{'status':'running','started_at':stamp(),'started_epoch':time.time()-1})
            cells=[]
            for home in ('h1','h2'):
                cell={'id':'fixed-seed-0','experiment':home,'method':'fixed_joint','seed':0};cells.append(cell)
                atomic_json(repo/'experiments'/home/'expt_v1/manifest.json',{'experiment':home,'cells':[cell],'config':{'cell_timeout_seconds':30}})
            dead=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);record=identity(dead.pid);dead.kill();dead.wait()
            out=repo/'experiments/h1/expt_v1/runtime/fixed-seed-0';atomic_json(out/'receipt.json',{'status':'running','process':record})
            child_script=root/'fake_cell.py'
            child_script.write_text('''import json,os,sys,pathlib\nr=pathlib.Path(os.environ['RECOVERY_DRILL_ROOT'])\nif '--initial' in sys.argv:\n p=r/'repo/experiments/00_program/runtime/initial-diagnostics.json';p.write_text(json.dumps({'complete':True,'evaluations':4}))\nelse:\n home=sys.argv[sys.argv.index('--experiment')+1];cell=sys.argv[sys.argv.index('--cell')+1]\n p=r/'repo/experiments'/home/'expt_v1/runtime'/cell/'receipt.json';p.write_text(json.dumps({'status':'complete','full_evaluation_complete':True}))\n''')
            original_sleep=time.sleep
            with patch.dict(os.environ,{'RECOVERY_DRILL_ROOT':str(root)}),patch.object(sys,'argv',['supervisor.py','--recover','--device','cpu']),patch.multiple(supervisor,REPO=repo,RUN_ROOT=run,PROGRAM=program,HOMES=('h1','h2'),__file__=str(child_script)),patch('supervisor.validate_gates',return_value={'queue_timeout_seconds':120,'calibration':{'eval_timeout_seconds':20}}),patch('integrity.verify_cell_artifacts',return_value=True),patch('supervisor.publish_snapshot',return_value=True),patch('supervisor.progress'),patch('supervisor.register_child',side_effect=lambda pid=None,role='worker':common.register_child(pid,'recovery_drill_'+role)),patch('supervisor.time.sleep',side_effect=lambda t:original_sleep(min(t,.01))):
                supervisor.main()
            self.assertEqual(json.loads((out/'receipt.json').read_text())['status'],'interrupted')
            self.assertEqual(json.loads((repo/'experiments/h2/expt_v1/runtime/fixed-seed-0/receipt.json').read_text())['status'],'complete')
            self.assertTrue(json.loads((program/'runtime/initial-diagnostics.json').read_text())['complete'])
            state=json.loads((program/'runtime/supervisor.json').read_text());self.assertEqual(state['totals']['interrupted'],1);self.assertEqual(state['totals']['complete'],1)

if __name__=='__main__':unittest.main()
