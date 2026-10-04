"""One prospective derivative of existing prepared data; no download/preparation rerun."""
import json,re
from pathlib import Path
from common import PROGRAM,DATA,atomic_json,digest,register_child

def grams(text):
    words=re.findall(r'\w+',text.casefold())
    return {tuple(words[i:i+13]) for i in range(max(0,len(words)-12))}

def main():
    register_child(role='split_and_overlap_audit')
    path=PROGRAM/'data_manifest.json';manifest=json.loads(path.read_text())
    if manifest.get('prospective_audit'):raise RuntimeError('already derived; preserve existing outputs')
    atomic_json(PROGRAM/'runtime/pre-audit-data-manifest.json',manifest)
    from transformers import AutoTokenizer
    spec=manifest['models']['scratch'];tok=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],local_files_only=True)
    def read(name):
        with (PROGRAM/manifest['splits'][name]['file']).open() as f:return [json.loads(l) for l in f]
    def write(key,rows,name=None):
        target=DATA/((name or key)+'.jsonl')
        with target.open('w') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        manifest['splits'][key]={'file':str(target.relative_to(PROGRAM)),'rows':len(rows),'sha256':digest(target)}
    sealed=set()
    for name in ('gsm_test','arc_test'):
        for r in read(name):
            sealed|=grams(r['question'])
            for s in ([r['answer']] if name=='gsm_test' else r['choices']['text']):sealed|=grams(s)
    counts={};retained={}
    for name in ('gsm_train','gsm_validation','prefs_train','prefs_validation'):
        rows=[];removed=0
        for r in read(name):
            texts=[r['question'],r['answer']] if name.startswith('gsm') else [tok.decode(r['prompt_ids']),tok.decode(r['chosen_ids']),tok.decode(r['rejected_ids'])]
            if any(grams(t)&sealed for t in texts):removed+=1;continue
            if name.startswith('gsm'):
                rationale=r['answer'].rsplit('####',1)[0]
                r['answer_start']=len(tok.encode(' '+rationale+'####',add_special_tokens=False))
            rows.append(r)
        counts[name]=removed;retained[name]=rows
        if name.endswith('train'):write(name,rows,name+'_audited')
    # WikiText overlap detection uses raw decoded train blocks as well. The
    # boundary overlap includes adjacent blocks, so a 13-gram cannot evade the
    # audit merely by crossing a 512-token boundary.
    wiki=read('wiki_train');remove=set()
    for i,r in enumerate(wiki):
        text=tok.decode(r['ids']);joined=tok.decode((wiki[i-1]['ids'][-64:] if i else [])+r['ids'][:64])
        if grams(text)&sealed or grams(joined)&sealed:remove.add(i);remove.add(max(0,i-1))
    counts['wiki_train']=len(remove);write('wiki_train',[r for i,r in enumerate(wiki) if i not in remove],'wiki_train_audited')
    for family in ('gsm','prefs','wiki'):
        rows=retained.get(family+'_validation') or read(family+'_validation')
        # IDs were fixed before measurement. Parity creates disjoint controller
        # feedback and development/admission populations; test stays sealed.
        import hashlib
        controller=[];development=[]
        for r in rows:
            target=controller if int(hashlib.sha256(r['id'].encode()).hexdigest()[:8],16)%2==0 else development
            target.append(r)
        write(family+'_controller',controller);write(family+'_development',development)
    write('arc_development',read('arc_validation'))
    manifest['prospective_audit']={'source_manifest_sha256':digest(PROGRAM/'runtime/pre-audit-data-manifest.json'),'ngram_words':13,'removed_rows':counts,'test_labels_use':'content-only contamination exclusion, never scoring/selection','controller_and_development':'disjoint deterministic ID-hash parity','prepared_originals_preserved':True}
    atomic_json(path,manifest);print(json.dumps(manifest['prospective_audit'],indent=2))

if __name__=='__main__':main()
