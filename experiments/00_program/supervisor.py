"""Independent one-device queue: explicit gates, durable receipts, no model calls."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from common import PROGRAM,REPO,RUN_ROOT,HOMES,atomic_json,digest,event,identity,alive,progress,register_child,stamp,runtime_versions

def validate_gates():
    gate=json.loads((PROGRAM/'acceptance.json').read_text())
    if gate['status']!='accepted' or gate['review_model']!='claude-opus-5-5' or gate['review_effort']!='max':raise RuntimeError('required review pending')
    if gate['plan_status']!='reconciled' or not gate['implementation_findings_reconciled']:raise RuntimeError('unreconciled findings')
    freeze=json.loads((PROGRAM/'frozen_program.json').read_text())
    for name,expected in freeze['hashes'].items():
        if digest(REPO/name)!=expected:raise RuntimeError('frozen input changed: '+name)
    if not freeze['calibration']['fits_48_hours']:raise RuntimeError('full program does not fit planning ceiling')
    if not freeze.get('setup_landed_commit'):raise RuntimeError('reviewed setup not landed')
    from integrity import verify_prepared_inputs,verify_landed_content
    verify_landed_content(freeze['setup_landed_commit'],freeze['landed_source_hashes'])
    verify_landed_content('HEAD',{**freeze['hashes'],'experiments/00_program/frozen_program.json':digest(PROGRAM/'frozen_program.json')})
    verify_prepared_inputs()
    return freeze

def completion_valid(returncode,receipt):
    return returncode==0 and receipt.get('status')=='complete' and receipt.get('full_evaluation_complete') is True

def initial_diagnostics(config,device):
    from training import make_models
    from evaluation import evaluate
    import torch
    manifest=json.loads((PROGRAM/'data_manifest.json').read_text())
    results=[]
    statuses={f'{h}/seed-{s}':{'status':'pending'} for h,seeds in [(HOMES[0],[0,1,2]),(HOMES[1],[0])] for s in seeds}
    atomic_json(PROGRAM/'runtime/initial-status.json',statuses)
    for experiment,seeds in [(HOMES[0],[0,1,2]),(HOMES[1],[0])]:
        for seed in seeds:
            out=REPO/'experiments'/experiment/'expt_v1/runtime'/f'initial-seed-{seed}'
            key=f'{experiment}/seed-{seed}';statuses[key]={'status':'running','timestamp':stamp()};atomic_json(PROGRAM/'runtime/initial-status.json',statuses)
            try:
                model,reference,tok=make_models(experiment,seed,device,manifest)
                results.append(evaluate(model,reference,tok,device,config,out,{'experiment':experiment,'seed':seed,'stage':'untouched_initial'}))
                statuses[key]={'status':'complete','summary_sha256':digest(out/'summary.json')};del model,reference
            except Exception as exc:statuses[key]={'status':'failed','exception_type':type(exc).__name__,'error':str(exc)}
            atomic_json(PROGRAM/'runtime/initial-status.json',statuses)
            if device=='cuda':torch.cuda.empty_cache()
    atomic_json(PROGRAM/'runtime/initial-diagnostics.json',{'complete':len(results)==4 and all(r['complete'] for r in results),'evaluations':len(results),'statuses':statuses})

def run_cell(cell,config,device):
    import torch
    from training import Engine
    from evaluation import evaluate
    home=REPO/'experiments'/cell['experiment']/'expt_v1'
    out=home/'runtime'/cell['id'];out.mkdir(parents=True,exist_ok=True)
    receipt=out/'receipt.json'
    claim=out/'cell.lock'
    fd=os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
    if receipt.exists():raise RuntimeError('cell has existing receipt; automatic regeneration forbidden')
    checkpoint=home/'checkpoints'/cell['id']/'state.pt'
    bound={'cell':cell,'config':config,'frozen_program_sha256':digest(PROGRAM/'frozen_program.json')}
    start=time.monotonic();memory=torch.cuda.mem_get_info() if device=='cuda' else None
    if memory and memory[0]<40*1024**3:raise RuntimeError('insufficient free device memory at cell admission')
    engine=Engine(cell['experiment'],cell['method'],cell['seed'],config,device)
    atomic_json(receipt,{'status':'running','identity':bound,'started_at':stamp(),'process':identity(),'device_memory_free_total':memory})
    engine.initialize();engine.checkpoint(checkpoint,bound)
    milestones=[.25,.5,.75,1.]
    while engine.work.total<config['budget_tokens']:
        if time.monotonic()-start>config['train_timeout_seconds']:raise TimeoutError('frozen training infrastructure limit')
        diag=engine.advance();engine.wall_seconds=time.monotonic()-start
        event(out/'events.jsonl',kind='update',charged_total=engine.work.total,**diag)
        if not (RUN_ROOT/'first_training_update.json').exists():
            atomic_json(RUN_ROOT/'first_training_update.json',{'run_id':RUN_ROOT.name,'timestamp':stamp(),'cell':cell,'step':engine.step,'objective':diag['objective'],'charged_tokens':engine.work.total,'process':identity(),'frozen_program_sha256':bound['frozen_program_sha256'],'measured':True,'smoke':False})
        if engine.next_milestone<len(milestones) and engine.work.total/config['budget_tokens']>=milestones[engine.next_milestone]:
            losses=engine.validate();event(out/'events.jsonl',kind='validation_milestone',fraction=milestones[engine.next_milestone],losses=losses,charged_total=engine.work.total)
            engine.next_milestone+=1
        if engine.step%config['checkpoint_every']==0:
            engine.checkpoint(checkpoint,bound);event(out/'events.jsonl',kind='checkpoint',step=engine.step,charged_total=engine.work.total)
        progress(measured_training_started=True,coordinator_status='training_running',phase='training',cell=cell['id'],experiment=cell['experiment'],step=engine.step,charged_tokens=engine.work.total,trainer_process=identity())
    engine.stage='evaluation';engine.wall_seconds=time.monotonic()-start;engine.checkpoint(checkpoint,bound)
    training={'steps':engine.step,'charged_tokens':engine.work.total,'budget_tokens':config['budget_tokens'],'overshoot_tokens':engine.work.total-config['budget_tokens'],
              'work':engine.work.state_dict(),'wall_seconds':engine.wall_seconds,'controller_iterations':engine.controller_iterations,'partial_probe':engine.probe_state is not None,
              'scheduler':engine.scheduler.snapshot(),'initial_validation':engine.initial,'final_validation':engine.current,
              'peak_memory_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,'parameter_count':sum(p.numel() for p in engine.model.parameters())}
    training['estimated_parameter_operations']=2*training['parameter_count']*training['charged_tokens']
    training['environment']=runtime_versions()
    with (out/'events.jsonl').open() as f:updates=[json.loads(line) for line in f if '"kind": "update"' in line]
    production=[]
    for u in updates:
        production.extend([u['sft'],u['rl']] if u.get('objective')=='parallel' else [u])
    training['objective_updates']={o:sum(u['objective']==o for u in production) for o in ('pt','sft','dpo','rl','rpt','hybrid')}
    training['gradient_clipped_updates']=sum(u.get('gradient_clipped',False) for u in production)
    training['rl_diagnostics']={o:{'updates':sum(u['objective']==o for u in production),'zero_gradient_skips':sum(u.get('zero_gradient',False) for u in production if u['objective']==o),'reward_sum':sum(u.get('reward_sum',0) for u in production if u['objective']==o),'rollouts':sum(u.get('rollouts',0) for u in production if u['objective']==o),'truncated':sum(u.get('truncated',0) for u in production if u['objective']==o)} for o in ('rl','rpt','hybrid')}
    # Maximum one update plus before/after/milestone validation. All training
    # sequences and math rollouts are <=512 tokens.
    validation_bound=4*config['validation_rows']*512+config['validation_rows']*(384+config['max_new_tokens'])
    overshoot_bound=max(8*config['batch_size']*512,3*config['batch_size']*512+4*config['rl_prompts']*config['group_size']*512)+3*validation_bound
    training['overshoot_bound_tokens']=overshoot_bound
    if training['overshoot_tokens']>overshoot_bound:raise RuntimeError('charged work exceeds frozen one-unit overshoot bound')
    atomic_json(out/'training.json',training)
    progress(phase='terminal_evaluation',cell=cell['id'])
    result=evaluate(engine.model,engine.reference,engine.tokenizer,device,config,out/'evaluation',bound)
    engine.stage='complete';atomic_json(out/'stage.json',{'stage':'complete','checkpoint_stage':'evaluation','timestamp':stamp()})
    final={'status':'complete','identity':bound,'finished_at':stamp(),'training':training,'evaluation_summary_sha256':digest(out/'evaluation/summary.json'),'checkpoint_sha256':digest(checkpoint),'wall_seconds':time.monotonic()-start,'full_evaluation_complete':result['complete']}
    atomic_json(receipt,final)

def update_ledgers(manifests):
    totals={'complete':0,'failed':0,'pending':0,'running':0}
    for manifest in manifests:
        home=REPO/'experiments'/manifest['experiment'];lines=['# Pilot results', '', '**Status:** Full-set execution is in progress; missing metrics are unavailable, not zero.', '', '| Method | Seed | Status | Training work (million forward-equivalent tokens) | ARC-Easy accuracy | GSM8K exact match | WikiText NLL | Preference raw / implicit accuracy |', '|---|---:|---|---:|---:|---:|---:|---|']
        rows=[]
        for cell in manifest['cells']:
            out=home/'expt_v1/runtime'/cell['id'];p=out/'receipt.json';r=json.loads(p.read_text()) if p.exists() else {'status':'pending'}
            s=r['status'];totals[s]=totals.get(s,0)+1;summary=out/'evaluation/summary.json'
            row={'id':cell['id'],'method':cell['method'],'seed':cell['seed'],'status':s}
            if s=='complete' and summary.exists():
                e=json.loads(summary.read_text());row.update(training_mfet=r['training']['charged_tokens']/1e6,arc=e['arc']['accuracy_primary_normalized'],gsm=e['gsm']['exact_match'],wiki_nll=e['wiki']['nll'],preference_raw=e['preferences']['raw_accuracy'],preference_implicit=e['preferences']['implicit_accuracy'])
                def fmt(m):return f"{m['estimate']:.4f} [{m['interval95'][0]:.4f}, {m['interval95'][1]:.4f}]"
                lines.append(f"| {cell['method']} | {cell['seed']} | {s} | {row['training_mfet']:.3f} | {fmt(row['arc'])} | {fmt(row['gsm'])} | {e['wiki']['nll']:.4f} [{e['wiki']['nll_interval95'][0]:.4f}, {e['wiki']['nll_interval95'][1]:.4f}] | {fmt(row['preference_raw'])} / {fmt(row['preference_implicit'])} |")
            else:lines.append(f"| {cell['method']} | {cell['seed']} | {s} | unavailable | unavailable | unavailable | unavailable | unavailable |")
            rows.append(row)
        lines+=['','NLL = negative log likelihood; preference raw ordering is the preference endpoint, implicit ordering is diagnostic. Scratch primary: WikiText NLL; early-checkpoint primary: ARC-Easy normalized accuracy. Unassisted GSM8K remains a sparse secondary diagnostic.','', 'Intervals: 95% Wilson binary-item score, conditional on each trained model; p-val=n/a (no hypothesis test). Three seeds do not establish universal superiority. Aioli, CHORD (Controllable Harmonization of On- and Off-Policy Reinforcement Learning via Dynamic Weighting), and RPT (Reinforcement Pre-Training) are declared adaptations; exact CHERRY-RL identity remains unresolved.','',f"Complete: {sum(r['status']=='complete' for r in rows)}/24. See `expt_v1/PROTOCOL.md` and the program review reconciliation."]
        if all(r['status']=='pending' for r in rows):lines[2]='**Status:** Not admitted; review and resource gates remain pending. Missing metrics are unavailable, not zero.'
        elif all(r['status']=='complete' for r in rows):lines[2]='**Status:** All 24 training cells have full terminal evaluations; program analysis, initial diagnostics and publication are separately verified.'
        (home/'results.md').write_text('\n'.join(lines)+'\n');atomic_json(home/'expt_v1/cell_status.json',rows)
    return totals

def free_gpu():
    # Fresh shared-node check before each child. No reservation is inferred.
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=30)
    for row in rows.splitlines():
        index,name,memory,util=[x.strip() for x in row.split(',')]
        if 'A100' in name and int(memory)<1024 and int(util)<5:return index
    raise RuntimeError('no available A100 at immediate admission check; no contention')

def interrupted_receipt(out,prior):
    events=[]
    if (out/'events.jsonl').exists():
        with (out/'events.jsonl').open() as f:events=[json.loads(l) for l in f]
    checkpoints=[e for e in events if e.get('kind')=='checkpoint']
    prior.update(status='interrupted',failure_class='infrastructure_interruption',failure='dead non-quiescent child; retained work is not regenerated',finished_at=stamp(),last_event=events[-1] if events else None,last_checkpoint_step=checkpoints[-1]['step'] if checkpoints else 0)
    atomic_json(out/'receipt.json',prior)
    return prior

def publish_snapshot(final):
    try:
        from report import main as analyze
        from publish_results import publish
        analyze()
        manifests=[json.loads((REPO/'experiments'/h/'expt_v1/manifest.json').read_text()) for h in HOMES]
        totals={'complete':0,'failed':0,'pending':0,'running':0}
        for m in manifests:
            for c in m['cells']:
                p=REPO/'experiments'/m['experiment']/'expt_v1/runtime'/c['id']/'receipt.json'
                status=json.loads(p.read_text())['status'] if p.exists() else 'pending';totals[status]=totals.get(status,0)+1
        text='# Full-program status\n\n'
        text+=f"Verified cells: {totals['complete']}/48; failed: {totals['failed']}; interrupted: {totals.get('interrupted',0)}; pending: {totals['pending']}; unstarted at global bound: {totals.get('not_started_global_bound',0)}.\n\n"
        text+=('All frozen cells and untouched initial diagnostics have complete evaluations. ' if final else 'Full-program execution is incomplete. ')
        text+='Required plan and implementation findings are reconciled; see acceptance.json.\n'
        text+='Results and uncertainty are in each experiment folder; no universal-superiority claim.\n'
        (PROGRAM/'FINAL_STATUS.md').write_text(text)
        receipt=publish()
        if receipt:progress(publication_status='verified_incremental',latest_landed_commit=receipt['commit'])
        return True
    except Exception as exc:
        event(PROGRAM/'runtime/publication-errors.jsonl',kind='publication_pending',error=str(exc));progress(publication_status='pending',publication_error=str(exc))
        return False

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cell');parser.add_argument('--experiment');parser.add_argument('--initial',action='store_true');parser.add_argument('--recover',action='store_true');parser.add_argument('--device',default='cuda');args=parser.parse_args()
    register_child(role='training_cell' if args.cell else ('initial_evaluation' if args.initial else 'training_supervisor'))
    freeze=validate_gates()
    manifests=[json.loads((REPO/'experiments'/h/'expt_v1/manifest.json').read_text()) for h in HOMES]
    if args.initial:
        initial_diagnostics(manifests[0]['config'],args.device);return
    if args.cell:
        m=next(m for m in manifests if m['experiment']==args.experiment);cell=next(c for c in m['cells'] if c['id']==args.cell)
        config={**m['config'],**{k:cell[k] for k in ('train_timeout_seconds','cell_timeout_seconds') if k in cell}}
        out=REPO/'experiments'/cell['experiment']/'expt_v1/runtime'/cell['id'];code=1
        try:
            run_cell(cell,config,args.device);code=0
        except Exception as exc:
            path=out/'receipt.json';r=json.loads(path.read_text()) if path.exists() else {'identity':{'cell':cell}}
            r.update(status='failed',failure_class='budget_incomplete' if isinstance(exc,TimeoutError) else ('numerical_failure' if isinstance(exc,FloatingPointError) else 'deterministic_exception'),exception_type=type(exc).__name__,error=str(exc),stage='evaluation' if (out/'training.json').exists() else 'training',finished_at=stamp())
            atomic_json(path,r);raise
        finally:atomic_json(out/'process-exit.json',{'returncode':code,'process':identity(),'timestamp':stamp()})
        return
    with (RUN_ROOT/'training-owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=PROGRAM/'runtime/supervisor.json'
        if state_path.exists():
            if not args.recover:raise RuntimeError('existing supervisor state; use explicit --recover')
            state=json.loads(state_path.read_text())
            if state.get('finished_at'):raise RuntimeError('queue already finalized')
            # Keep the original queue clock; never reset a budget on recovery.
            if 'started_epoch' not in state:raise RuntimeError('missing original queue clock')
        else:state={'status':'running','started_at':stamp(),'started_epoch':time.time()}
        state['process']=identity();atomic_json(state_path,state)
        atomic_json(RUN_ROOT/'driver_identity.json',{'run_id':RUN_ROOT.name,'timestamp':stamp(),'process':identity(),'status':'supervisor_running'})
        start=time.monotonic()-(time.time()-state['started_epoch'])
        for manifest in manifests:
            for cell in manifest['cells']:
                out=REPO/'experiments'/manifest['experiment']/'expt_v1/runtime'/cell['id'];out.mkdir(parents=True,exist_ok=True)
                prior=json.loads((out/'receipt.json').read_text()) if (out/'receipt.json').exists() else {}
                if prior.get('status') in ('complete','failed','interrupted','not_started_global_bound'):continue
                if time.monotonic()-start>freeze['queue_timeout_seconds']:
                    atomic_json(out/'receipt.json',{'status':'not_started_global_bound','identity':{'cell':cell},'timestamp':stamp()});continue
                child=None;record=prior.get('process')
                if record and alive(record):
                    child_id=record;cell_start=time.monotonic()-(time.time()-__import__('datetime').datetime.fromisoformat(prior['started_at']).timestamp())
                    event(out/'supervisor-events.jsonl',kind='healthy_child_adopted',process=record)
                elif prior:
                    interrupted_receipt(out,prior)
                    continue
                else:
                    # Protect the spawn-to-receipt race after a supervisor crash.
                    healthy=[r for r in json.loads((RUN_ROOT/'children.json').read_text()) if r.get('role')=='training_cell' and alive(r)]
                    if healthy:raise RuntimeError('unmatched healthy training child; no duplicate launch')
                    env=os.environ.copy()
                    if args.device=='cuda':env['CUDA_VISIBLE_DEVICES']=free_gpu()
                    with (out/'process.log').open('a') as log:
                        child=subprocess.Popen([sys.executable,__file__,'--cell',cell['id'],'--experiment',manifest['experiment'],'--device',args.device],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                    child_id=register_child(child.pid,'training_cell');cell_start=time.monotonic()
                while alive(child_id) and (child is None or child.poll() is None):
                    atomic_json(RUN_ROOT/'watchdog.json',{'run_id':RUN_ROOT.name,'timestamp':stamp(),'status':'training_alive','process':identity(),'child':child_id,'cell':cell['id'],'child_alive':True})
                    if time.monotonic()-cell_start>cell.get('cell_timeout_seconds',manifest['config']['cell_timeout_seconds']):
                        event(out/'failures.jsonl',kind='timeout',reason='cell_timeout')
                        import signal
                        os.killpg(child_id['pid'],signal.SIGTERM)
                        deadline=time.monotonic()+30
                        while alive(child_id) and time.monotonic()<deadline:time.sleep(1)
                        if alive(child_id):os.killpg(child_id['pid'],signal.SIGKILL)
                        break
                    time.sleep(15)
                if child is not None:child.wait()
                exit_file=out/'process-exit.json'
                exit_code=child.returncode if child is not None else (json.loads(exit_file.read_text()).get('returncode') if exit_file.exists() else None)
                p=out/'receipt.json';r=json.loads(p.read_text()) if p.exists() else {}
                valid=completion_valid(exit_code,r)
                if valid:
                    try:
                        from integrity import verify_cell_artifacts
                        valid=verify_cell_artifacts(out,cell)
                    except Exception as exc:
                        valid=False;event(out/'failures.jsonl',kind='artifact_verification_failure',error=str(exc))
                if not valid:
                    event(out/'failures.jsonl',kind='infrastructure_failure',returncode=exit_code,prior_status=r.get('status'))
                    r.update(status='failed',returncode=exit_code,failure='See retained process.log; no automatic regeneration',finished_at=stamp());r.setdefault('failure_class','infrastructure_interruption' if exit_code is None or exit_code<0 else 'incomplete_artifacts');atomic_json(p,r)
                totals=update_ledgers(manifests);progress(**{f'cells_{k}':v for k,v in totals.items()},phase='queue',coordinator_status='training_running')
                publish_snapshot(False)
        totals=update_ledgers(manifests);complete=totals['complete']==48
        # Untouched diagnostics occur only after fixed cell training/evaluation;
        # they cannot select checkpoint age or modify the frozen program.
        if True:  # All four initial evaluations remain declared even after failed cells.
            with (PROGRAM/'runtime/initial-process.log').open('a') as log:
                env=os.environ.copy()
                if args.device=='cuda':env['CUDA_VISIBLE_DEVICES']=free_gpu()
                child=subprocess.Popen([sys.executable,__file__,'--initial','--device',args.device],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                child_id=register_child(child.pid,'initial_evaluation')
                initial_start=time.monotonic()
                while child.poll() is None:
                    atomic_json(RUN_ROOT/'watchdog.json',{'run_id':RUN_ROOT.name,'timestamp':stamp(),'status':'initial_evaluation_alive','child':child_id,'child_alive':alive(child_id)})
                    if time.monotonic()-initial_start>4*(freeze['calibration']['eval_timeout_seconds']+60):
                        child.terminate()
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        break
                    time.sleep(15)
                summary=PROGRAM/'runtime/initial-diagnostics.json'
                complete=complete and child.returncode==0 and summary.exists() and json.loads(summary.read_text()).get('complete') is True
        publication=publish_snapshot(complete)
        fully_complete=complete and publication
        state.update(status='complete' if fully_complete else 'partial',finished_at=stamp(),totals=totals,wall_seconds=time.monotonic()-start)
        atomic_json(state_path,state)
        progress(coordinator_status='complete' if fully_complete else 'blocked',phase='complete' if fully_complete else 'remaining_work_pending',publication_status='verified' if publication else 'pending',**{f'cells_{k}':v for k,v in totals.items()})
        atomic_json(RUN_ROOT/'driver_identity.json',{'run_id':RUN_ROOT.name,'timestamp':stamp(),'process':identity(),'status':'finished'})
        atomic_json(RUN_ROOT/'watchdog.json',{'run_id':RUN_ROOT.name,'timestamp':stamp(),'status':'complete' if fully_complete else 'partial','totals':totals})

if __name__=='__main__':
    try:main()
    except Exception as exc:
        # A failed coordinator must never label a surviving child stopped.
        rows=json.loads((RUN_ROOT/'children.json').read_text()) if (RUN_ROOT/'children.json').exists() else []
        living=[r for r in rows if r.get('role')=='training_cell' and alive(r) and r['pid']!=os.getpid()]
        progress(coordinator_status='training_running' if living else 'blocked',phase='supervisor_error',supervisor_error=str(exc))
        atomic_json(RUN_ROOT/'watchdog.json',{'run_id':RUN_ROOT.name,'timestamp':stamp(),'status':'surviving_training_child' if living else 'blocked','living_children':living,'error':str(exc)})
        raise
