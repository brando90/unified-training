"""Additional prospective substring audit of preserved public data; no scoring."""
import collections,json,re
from common import PROGRAM,DATA,atomic_json,digest
from training import load_data

def norm(text):return ' '.join(re.findall(r'\w+',text.casefold()))
def main():
    from transformers import AutoTokenizer
    manifest=json.loads((PROGRAM/'data_manifest.json').read_text())
    if manifest.get('substring_audit'):raise RuntimeError('audit already recorded; no repeated derivation')
    spec=manifest['models']['scratch'];tok=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],local_files_only=True)
    questions={norm(r['question']) for split in ('gsm_test','arc_test') for r in load_data(split) if len(norm(r['question']))>=40}
    index=collections.defaultdict(list)
    for q in questions:
        w=q.split();index[tuple(w[:min(5,len(w))])].append(q)
    lengths={len(k) for k in index}
    def contaminated(text):
        text=norm(text);words=text.split()
        for n in lengths:
            for i in range(len(words)-n+1):
                for q in index.get(tuple(words[i:i+n]),[]):
                    if ' '+q+' ' in ' '+text+' ':return True
        return False
    pref_test=[norm(tok.decode(r['prompt_ids'])) for r in load_data('prefs_test')]
    counts={};sha={}
    for split in ('gsm_train','prefs_train','wiki_train'):
        rows=load_data(split);kept=[];removed=[]
        for i,r in enumerate(rows):
            if split=='gsm_train':texts=[r['question'],r['answer']];cross=any(' '+norm(r['question'])+' ' in ' '+p+' ' for p in pref_test)
            elif split=='prefs_train':texts=[tok.decode(r[k]) for k in ('prompt_ids','chosen_ids','rejected_ids')];cross=False
            else:texts=[tok.decode((rows[i-1]['ids'] if i else [])+r['ids'])];cross=False
            if cross or any(contaminated(t) for t in texts):removed.append(r['id'])
            else:kept.append(r)
        counts[split]=len(removed);sha[split]=__import__('hashlib').sha256(json.dumps(removed).encode()).hexdigest()
        if removed:
            p=DATA/(split+'_substring_audited.jsonl')
            with p.open('x') as f:
                for r in kept:f.write(json.dumps(r,ensure_ascii=False)+'\n')
            manifest['splits'][split]={'file':str(p.relative_to(PROGRAM)),'rows':len(kept),'sha256':digest(p)}
    manifest['substring_audit']={'minimum_normalized_question_characters':40,'removed_rows':counts,'excluded_ids_digest':sha,'gsm_train_checked_against_preference_test_prompts':True,'test_scores_used':False}
    atomic_json(PROGRAM/'data_manifest.json',manifest);print(json.dumps(manifest['substring_audit']))
if __name__=='__main__':main()
