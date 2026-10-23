# Paired-difference tables for results.md. Run from this directory after: tar xzf p05_results.tgz (needs collect/, runs/). Writes tables.json to the absolute path below; edit it.
import json, glob, math, statistics as st, sys
R='results_final/'
d=json.load(open(R+'collect/collect.json'))
cells={c['cell']:c for c in d['cells']}
sizes=['14m','70m','160m']; meths=['sequential','fixed_joint','smooth_joint']; seeds=[1,2,3]
def key(s,m,k): 
    v='v1' if s=='160m' else 'v2'
    return cells[f'{v}/{s}/{m}_s{k}']
def lc(s,m,k,ck):
    v='v1' if s=='160m' else 'v2'
    j=json.load(open(R+f'runs/{v}/{s}/{m}_s{k}/eval/lc_{ck}/summary.json'))
    return j['wikitext']['nll_per_token'], j['arc_easy']['acc'], j['lc_gsm8k']['acc']
T=4.303
def ms(x): 
    m=st.mean(x); sd=st.stdev(x); return m,sd
out=[]
P=out.append
# status table
P('| cell | job ids (first attempt, resumed run) | work charged | overshoot | observed PT tokens | fp16 skipped steps | train+eval wall h | WikiText NLL | ARC-E acc | GSM8K acc [95% Wilson] (n=1319) | UltraFeedback pref acc (n=1985) |')
P('|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|')
for s in ['160m','14m','70m']:
  for k in seeds:
    for m in meths:
      c=key(s,m,k)
      w=c['gsm8k_wilson95@100']
      P(f"| {c['cell']} | {', '.join(c['slurm_job_ids'])} | {c['work_total']:,} | {c['work_overshoot']:,.0f} | {c['observed_pt_tokens']:,} | {c['fp16_skipped_steps']} | {c['wall_h']:.2f} | {c['wikitext_nll@100']:.3f} | {c['arc_easy_acc@100']:.3f} | {c['gsm8k_acc@100']:.4f} [{w[0]:.4f}, {w[1]:.4f}] | {c['pref_acc@100']:.3f} |")
tab1='\n'.join(out); out.clear()
# means
P('| size | method | WikiText NLL mean (SD) [per-seed] | ARC-E acc | GSM8K acc | pref acc | observed PT tokens (M) |')
P('|---|---|---|---|---|---|---:|')
for s in sizes:
  for m in meths:
    cs=[key(s,m,k) for k in seeds]
    f=lambda n: [c[n] for c in cs]
    nll=f('wikitext_nll@100'); a=ms(nll)
    P(f"| {s} | {m} | {a[0]:.3f} ({a[1]:.3f}) [{', '.join(f'{x:.2f}' for x in nll)}] | {st.mean(f('arc_easy_acc@100')):.3f} | {st.mean(f('gsm8k_acc@100')):.4f} | {st.mean(f('pref_acc@100')):.3f} | {st.mean(f('observed_pt_tokens'))/1e6:.1f} |")
tab2='\n'.join(out); out.clear()
# paired
P('| size | contrast | metric | seed 1, 2, 3 | mean | range | 95% t-interval of the mean (df=2, descriptive) | seeds favouring joint | protocol verdict |')
P('|---|---|---|---|---:|---|---|---|---|')
mets=[('wikitext_nll@100','WikiText NLL (primary; negative = joint better)',-1,4),('arc_easy_acc@100','ARC-E acc (positive = joint better)',1,4),('gsm8k_acc@100','GSM8K acc (positive = joint better)',1,4),('pref_acc@100','pref acc (positive = joint better)',1,4)]
summ={}
for s in sizes:
  for j in ['fixed_joint','smooth_joint']:
    for mk,mn,sign,nd in mets:
      g=[key(s,j,k)[mk]-key(s,'sequential',k)[mk] for k in seeds]
      m,sd=ms(g); h=T*sd/math.sqrt(3)
      fav=sum(1 for x in g if x*sign>0)
      verd=''
      if mk.startswith('wikitext'): verd='joint better' if fav==3 else ('joint worse' if fav==0 else 'mixed')
      summ[(s,j,mk)]=(m,h,fav,g)
      P(f"| {s} | {j} - sequential | {mn} | {', '.join(f'{x:+.{nd}f}' for x in g)} | {m:+.{nd}f} | [{min(g):+.{nd}f}, {max(g):+.{nd}f}] | [{m-h:+.{nd}f}, {m+h:+.{nd}f}] | {fav}/3 | {verd} |")
tab3='\n'.join(out); out.clear()
# size readout
P('| size (params) | fixed_joint - sequential NLL (mean, range) | smooth_joint - sequential NLL (mean, range) | sequential NLL mean | fixed_joint NLL mean | smooth_joint NLL mean |')
P('|---|---|---|---|---|---|')
for s in sizes:
  a=summ[(s,'fixed_joint','wikitext_nll@100')];b=summ[(s,'smooth_joint','wikitext_nll@100')]
  q=lambda m: st.mean([key(s,m,k)['wikitext_nll@100'] for k in seeds])
  P(f"| {s} | {a[0]:+.2f} ({min(a[3]):+.2f} to {max(a[3]):+.2f}) | {b[0]:+.2f} ({min(b[3]):+.2f} to {max(b[3]):+.2f}) | {q('sequential'):.2f} | {q('fixed_joint'):.3f} | {q('smooth_joint'):.3f} |")
tab4='\n'.join(out); out.clear()
# learning curves
P('| size | method | NLL at 0% | 25% | 50% | 75% | 100% (mean of 3 seeds) |')
P('|---|---|---:|---:|---:|---:|---:|')
for s in sizes:
  for m in meths:
    row=[st.mean([lc(s,m,k,ck)[0] for k in seeds]) for ck in ['000','025','050','075','100']]
    P(f"| {s} | {m} | "+' | '.join(f'{x:.3f}' for x in row)+' |')
tab5='\n'.join(out); out.clear()
# pooled gsm8k
P('| size | method | GSM8K correct / 3 x 1,319 |'); P('|---|---|---|')
for s in sizes:
  for m in meths:
    n=sum(round(key(s,m,k)['gsm8k_acc@100']*1319) for k in seeds)
    P(f"| {s} | {m} | {n} / 3957 |")
tab6='\n'.join(out)
json.dump(dict(t1=tab1,t2=tab2,t3=tab3,t4=tab4,t5=tab5,t6=tab6),open('/var/lib/agent/scratch/at583/tables.json','w'))
for t in (tab2,tab4,tab5,tab6): print(t,'\n')
print(summ[('14m','fixed_joint','wikitext_nll@100')])
tot=sum(c['wall_h'] for c in cells.values()); print('wall h total',tot, 'work min/max',min(c['work_total'] for c in cells.values()),max(c['work_total'] for c in cells.values()), 'pt min',min(c['observed_pt_tokens'] for c in cells.values()))
print(sorted(set(c['device'] for c in cells.values())), sum(1 for c in cells.values() if c['reached_pt_target']))
