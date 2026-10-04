"""Pin official releases and prepare real, audited data without allocating a GPU.

Test labels are only serialized for terminal evaluation; overlap audits use text
identities. No scores or test outcomes select training examples or hyperparameters.
"""
from __future__ import annotations
import collections
import hashlib
import json
import re
from pathlib import Path
from common import DATA, PROGRAM, atomic_json, digest, register_child, stamp

SOURCES = {
    'wiki': ('Salesforce/wikitext', 'wikitext-103-raw-v1'),
    'gsm': ('openai/gsm8k', 'main'),
    'prefs': ('HuggingFaceH4/ultrafeedback_binarized', None),
    'arc': ('allenai/ai2_arc', 'ARC-Easy'),
}
MAX_LENGTH = 512

def normalized(s):
    return re.sub(r'\s+', ' ', s).strip().casefold()

def text_hash(s):
    return hashlib.sha256(normalized(s).encode()).hexdigest()

def prompt(q):
    return f'Question: {q.strip()}\nAnswer:'

def main():
    register_child(role='data_preparation')
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from transformers import AutoTokenizer
    api = HfApi(); DATA.mkdir(parents=True, exist_ok=True)
    records = {'created': stamp(), 'sources': {}, 'models': {}, 'filters': {}, 'splits': {}}
    for key, name, revision in [('scratch', 'EleutherAI/pythia-70m', 'main'),
                                ('early', 'EleutherAI/pythia-160m', 'step10000')]:
        info = api.model_info(name, revision=revision)
        local = snapshot_download(name, revision=info.sha, allow_patterns=['config.json', 'tokenizer*', 'special_tokens_map.json', '*.safetensors', 'pytorch_model.bin'],
                                  ignore_patterns=['pytorch_model.bin'] if any(s.rfilename.endswith('.safetensors') for s in info.siblings) else None)
        records['models'][key] = {'id': name, 'requested_revision': revision, 'revision': info.sha,
                                  'license': 'apache-2.0', 'files': {p.name: digest(p) for p in Path(local).iterdir() if p.is_file()}}
        print('MODEL', key, info.sha, flush=True)
    tok = AutoTokenizer.from_pretrained(records['models']['scratch']['id'], revision=records['models']['scratch']['revision'])
    # Pythia models must share the exact tokenizer, not merely a vocabulary size.
    tok2 = AutoTokenizer.from_pretrained(records['models']['early']['id'], revision=records['models']['early']['revision'])
    assert tok.get_vocab() == tok2.get_vocab()
    loaded = {}
    for key, (name, config) in SOURCES.items():
        info = api.dataset_info(name)
        card = info.card_data.to_dict() if info.card_data else {}
        records['sources'][key] = {'id': name, 'config': config, 'revision': info.sha, 'license': card.get('license', 'see source card'), 'url': f'https://huggingface.co/datasets/{name}/tree/{info.sha}'}
        try:
            card_path = hf_hub_download(name, 'README.md', repo_type='dataset', revision=info.sha)
            records['sources'][key]['card_sha256'] = digest(card_path)
        except Exception as e:
            records['sources'][key]['card_error'] = type(e).__name__
        loaded[key] = load_dataset(name, config, revision=info.sha)
        records['sources'][key]['raw_rows'] = {k: len(v) for k, v in loaded[key].items()}
        print('DATASET', key, records['sources'][key]['raw_rows'], flush=True)
    sealed_prompts = {text_hash(r['question']) for d in ('gsm', 'arc') for r in loaded[d]['test']}
    validation_prompts = {text_hash(r['question']) for r in loaded['arc']['validation']}
    # Full official benchmark rows; do not filter evaluation by model fit.
    out = {'gsm_test': [], 'arc_test': [], 'arc_validation': []}
    for row in loaded['gsm']['test']:
        out['gsm_test'].append({'id': text_hash(row['question']), 'question': row['question'], 'prompt': prompt(row['question']), 'answer': row['answer'], 'prompt_ids': tok.encode(prompt(row['question']), add_special_tokens=False)})
    for row in loaded['arc']['test']:
        q = prompt(row['question']); ids = tok.encode(q, add_special_tokens=False)
        out['arc_test'].append({'id': row['id'], 'question': row['question'], 'prompt': q, 'prompt_ids': ids, 'choices': row['choices'], 'answerKey': row['answerKey']})
    for row in loaded['arc']['validation']:
        q=prompt(row['question'])
        out['arc_validation'].append({'id':row['id'],'question':row['question'],'prompt':q,'prompt_ids':tok.encode(q,add_special_tokens=False),'choices':row['choices'],'answerKey':row['answerKey']})
    assert len(out['gsm_test']) == 1319 and len(out['arc_test']) == 2376
    exclusions = collections.Counter(); seen = set(); out['gsm_train'] = []; out['gsm_validation'] = []
    for row in loaded['gsm']['train']:
        h = text_hash(row['question'])
        if h in sealed_prompts or h in validation_prompts: exclusions['benchmark_prompt_overlap'] += 1; continue
        if h in seen: exclusions['duplicate'] += 1; continue
        seen.add(h)
        p = tok.encode(prompt(row['question']), add_special_tokens=False)
        a = tok.encode(' ' + row['answer'], add_special_tokens=False) + [tok.eos_token_id]
        if len(p) > 256 or len(p) + len(a) > MAX_LENGTH: exclusions['full_response_overlength'] += 1; continue
        split = 'validation' if int(h[:8], 16) % 10 == 0 else 'train'
        out['gsm_' + split].append({'id': h, 'question': row['question'], 'answer': row['answer'], 'prompt_ids': p, 'response_ids': a})
    records['filters']['gsm'] = dict(exclusions)
    # Preferences use full assistant responses and single-turn conversations.
    exclusions = collections.Counter(); out['prefs_train'] = []; out['prefs_validation'] = []; out['prefs_test'] = []
    gsm_val_hashes = {x['id'] for x in out['gsm_validation']}
    pref_test_hashes = {text_hash(r['prompt']) for r in loaded['prefs']['test_prefs']}
    seen = set()
    for source, target in [('test_prefs', 'test'), ('train_prefs', 'train')]:
        for row in loaded['prefs'][source]:
            h = text_hash(row['prompt'])
            if h in sealed_prompts | validation_prompts | gsm_val_hashes: exclusions['prompt_overlap'] += 1; continue
            if source == 'train_prefs' and h in pref_test_hashes: exclusions['official_test_overlap'] += 1; continue
            if h in seen: exclusions['duplicate'] += 1; continue
            seen.add(h)
            chosen, rejected = row['chosen'], row['rejected']
            if len(chosen) != 2 or len(rejected) != 2 or chosen[-1]['role'] != 'assistant' or rejected[-1]['role'] != 'assistant': exclusions['not_single_turn'] += 1; continue
            p = tok.encode(prompt(row['prompt']), add_special_tokens=False)
            c = tok.encode(' ' + chosen[-1]['content'], add_special_tokens=False) + [tok.eos_token_id]
            r = tok.encode(' ' + rejected[-1]['content'], add_special_tokens=False) + [tok.eos_token_id]
            if not chosen[-1]['content'].strip() or not rejected[-1]['content'].strip() or c == r: exclusions['empty_or_identical'] += 1; continue
            if max(len(p)+len(c), len(p)+len(r)) > MAX_LENGTH: exclusions['full_pair_overlength'] += 1; continue
            split = target if target == 'test' else ('validation' if int(h[:8], 16) % 10 == 0 else 'train')
            out['prefs_' + split].append({'id': h, 'prompt_ids': p, 'chosen_ids': c, 'rejected_ids': r})
    for split, cap in [('train',4096), ('validation',256), ('test',256)]:
        key = 'prefs_' + split; out[key].sort(key=lambda x:x['id']); exclusions['cap_' + split] = max(0,len(out[key])-cap); out[key] = out[key][:cap]
    records['filters']['prefs'] = dict(exclusions)
    # Remove remaining cross-source train/validation prompt identities.
    val_ids = {r['id'] for k in ('gsm_validation', 'prefs_validation', 'prefs_test') for r in out[k]}
    for key in ('gsm_train', 'prefs_train'):
        before = len(out[key]); out[key] = [r for r in out[key] if r['id'] not in val_ids]
        records['filters'][key + '_cross_source'] = before-len(out[key])
    # Corpus line deduplication across official train/validation/test. Chunking
    # has disjoint targets, EOS between lines, no padded tokens in the denominator.
    wiki = loaded['wiki']; excluded_lines = {text_hash(x['text']) for split in ('validation','test') for x in wiki[split] if x['text'].strip()}
    excluded_lines |= sealed_prompts | val_ids
    test_lines = {text_hash(x['text']) for x in wiki['test'] if x['text'].strip()}
    for split in ('train','validation','test'):
        rows = []; token_buffer = []; count = collections.Counter(); seen_lines = set()
        for row in wiki[split]:
            text = row['text']; h = text_hash(text)
            if not text.strip(): count['empty'] += 1; continue
            if split=='train' and h in excluded_lines or split=='validation' and h in test_lines: count['heldout_overlap'] += 1; continue
            if split=='train' and h in seen_lines: count['duplicate'] += 1; continue
            seen_lines.add(h)
            token_buffer.extend(tok.encode(text, add_special_tokens=False) + [tok.eos_token_id])
            while len(token_buffer) >= MAX_LENGTH:
                ids=token_buffer[:MAX_LENGTH]; del token_buffer[:MAX_LENGTH]
                rows.append({'id':f'{split}-{len(rows)}','ids':ids})
            if split=='train' and len(rows)>=8192: count['prospective_token_pool_cap'] = 8192*MAX_LENGTH; break
        if len(token_buffer)>1 and not (split=='train' and len(rows)>=8192): rows.append({'id':f'{split}-{len(rows)}','ids':token_buffer})
        out['wiki_'+split]=rows; records['filters']['wiki_'+split]=dict(count)
    for key, rows in out.items():
        assert rows, key
        path=DATA/(key+'.jsonl')
        with path.open('w') as f:
            for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
        records['splits'][key]={'rows':len(rows),'sha256':digest(path),'file':str(path.relative_to(PROGRAM))}
    records['overlap_policy'] = 'Normalized prompt/line hashes; official test excluded before hash validation split; exact text only, semantic contamination unknown. SFT and RL deliberately reuse GSM training pool.'
    records['max_length']=MAX_LENGTH
    atomic_json(PROGRAM/'data_manifest.json',records)
    print('PREPARATION_COMPLETE',json.dumps({k:v['rows'] for k,v in records['splits'].items()}),flush=True)

if __name__ == '__main__': main()
