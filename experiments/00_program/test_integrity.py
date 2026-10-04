"""Prepared-data integrity and deterministic process/finalization checks."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from common import PROGRAM,DATA,digest,identity,alive,atomic_json
from training import load_data
from prepare_data import text_hash
from supervisor import completion_valid,interrupted_receipt

class IntegrityTests(unittest.TestCase):
    def test_unicode_jsonl_separator_is_not_a_record_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            row={'question':'first\u2028second\u0085third'}
            (Path(td)/'example.jsonl').write_text(json.dumps(row,ensure_ascii=False)+'\n')
            with patch('training.DATA',Path(td)):self.assertEqual(load_data('example'),[row])
    @unittest.skipUnless((PROGRAM/'data_manifest.json').exists(),'real preparation not available')
    def test_real_prepared_hashes_and_counts(self):
        manifest=json.loads((PROGRAM/'data_manifest.json').read_text())
        for name,spec in manifest['splits'].items():
            path=PROGRAM/spec['file'];self.assertEqual(digest(path),spec['sha256']);self.assertEqual(len(load_data(name)),spec['rows'])
        self.assertEqual(manifest['splits']['gsm_test']['rows'],1319);self.assertEqual(manifest['splits']['arc_test']['rows'],2376)
    @unittest.skipUnless((PROGRAM/'data_manifest.json').exists(),'real preparation not available')
    def test_real_prompt_split_integrity(self):
        sealed={text_hash(r['question']) for k in ('gsm_test','arc_test','arc_validation') for r in load_data(k)}
        train={r['id'] for k in ('gsm_train','prefs_train') for r in load_data(k)}
        val={r['id'] for k in ('gsm_validation','prefs_validation','prefs_test') for r in load_data(k)}
        self.assertFalse(train & sealed);self.assertFalse(train & val);self.assertFalse(val & sealed)
        for row in load_data('prefs_train'):
            self.assertLessEqual(len(row['prompt_ids'])+len(row['chosen_ids']),512)
            self.assertLessEqual(len(row['prompt_ids'])+len(row['rejected_ids']),512)
            self.assertNotEqual(row['chosen_ids'],row['rejected_ids'])
    def test_actual_dummy_failure_and_zero_exit_are_not_completion(self):
        bad=subprocess.run([sys.executable,'-c','raise SystemExit(7)'])
        good=subprocess.run([sys.executable,'-c','pass'])
        self.assertFalse(completion_valid(bad.returncode,{}));self.assertFalse(completion_valid(good.returncode,{}))
        self.assertFalse(completion_valid(0,{'status':'complete','full_evaluation_complete':False}))
        self.assertTrue(completion_valid(0,{'status':'complete','full_evaluation_complete':True}))
    def test_actual_living_child_identity(self):
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(2)'])
        try:
            record=identity(child.pid);self.assertTrue(alive(record));record['boot_id']='incorrect';self.assertFalse(alive(record))
        finally:child.terminate();child.wait()
    def test_landed_content_rejects_dirty_source(self):
        from integrity import verify_landed_content
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'source.py').write_text('x=1\n')
            for args in (['init','-q'],['add','source.py'],['-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture']):subprocess.run(['git',*args],cwd=root,check=True)
            hashes={'source.py':digest(root/'source.py')};self.assertTrue(verify_landed_content('HEAD',hashes,root))
            (root/'source.py').write_text('x=2\n')
            with self.assertRaisesRegex(RuntimeError,'landed content mismatch'):verify_landed_content('HEAD',hashes,root)
    def test_recovery_records_killed_child_and_preserves_checkpoint_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);(out/'events.jsonl').write_text(json.dumps({'kind':'checkpoint','step':32})+'\n'+json.dumps({'kind':'update','step':33})+'\n')
            child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
            record=identity(child.pid);self.assertTrue(alive(record));child.kill();child.wait()
            receipt=interrupted_receipt(out,{'status':'running','process':record})
            self.assertEqual(receipt['status'],'interrupted');self.assertEqual(receipt['last_checkpoint_step'],32);self.assertEqual(receipt['last_event']['step'],33)
    def test_controller_development_disjoint(self):
        for family in ('gsm','prefs','wiki'):
            self.assertFalse({r['id'] for r in load_data(family+'_controller')} & {r['id'] for r in load_data(family+'_development')})

if __name__=='__main__':unittest.main()
