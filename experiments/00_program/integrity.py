"""Content and completion checks independent of a child process's exit status."""
import json,hashlib,subprocess
from pathlib import Path
from common import PROGRAM,REPO,digest

def verify_landed_content(commit,hashes,repo=REPO):
    for name,expected in hashes.items():
        content=subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=repo,timeout=30)
        if hashlib.sha256(content).hexdigest()!=expected or digest(repo/name)!=expected:raise RuntimeError('landed content mismatch: '+name)
    return True

def verify_prepared_inputs():
    from huggingface_hub import hf_hub_download
    manifest=json.loads((PROGRAM/'data_manifest.json').read_text())
    for name,spec in manifest['splits'].items():
        path=PROGRAM/spec['file']
        if digest(path)!=spec['sha256']:raise RuntimeError('prepared split hash mismatch: '+name)
        with path.open() as f:
            n=sum(1 for line in f if line.strip())
        if n!=spec['rows']:raise RuntimeError('prepared split count mismatch: '+name)
    for model in manifest['models'].values():
        for filename,sha in model['files'].items():
            path=hf_hub_download(model['id'],filename,revision=model['revision'],local_files_only=True)
            if digest(path)!=sha:raise RuntimeError('model/tokenizer file hash mismatch: '+filename)
    return manifest

def verify_cell_artifacts(out,cell):
    out=Path(out);receipt=json.loads((out/'receipt.json').read_text());summary=out/'evaluation/summary.json'
    if receipt.get('identity',{}).get('cell')!=cell:raise RuntimeError('cell identity mismatch')
    if digest(summary)!=receipt.get('evaluation_summary_sha256'):raise RuntimeError('evaluation summary hash mismatch')
    e=json.loads(summary.read_text())
    data=json.loads((PROGRAM/'data_manifest.json').read_text())
    checks=[('wiki','wiki_test','blocks'),('preferences','prefs_test','rows'),('arc','arc_test','rows'),('gsm','gsm_test','rows')]
    for key,split,count in checks:
        if e[key][count]!=data['splits'][split]['rows']:raise RuntimeError('evaluation denominator mismatch: '+key)
        with (PROGRAM/data['splits'][split]['file']).open() as f:expected={json.loads(l)['id'] for l in f}
        with (out/'evaluation'/f'{key}.jsonl').open() as f:rows=[json.loads(l) for l in f]
        observed=[r['id'] for r in rows]
        if len(observed)!=len(expected) or set(observed)!=expected:raise RuntimeError('missing or duplicate evaluation items: '+key)
    checkpoint=REPO/'experiments'/cell['experiment']/'expt_v1/checkpoints'/cell['id']/'state.pt'
    if digest(checkpoint)!=receipt.get('checkpoint_sha256'):raise RuntimeError('checkpoint hash mismatch')
    return e.get('complete') is True
