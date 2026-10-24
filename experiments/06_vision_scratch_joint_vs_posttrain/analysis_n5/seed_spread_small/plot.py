import json,glob
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R={}
for f in glob.glob('raw/runs/small/*/seed*/curve.jsonl'):
    p=f.split('/'); R[(p[3],int(p[4][4:]))]=[json.loads(l) for l in open(f)]
col={1:'#4c78a8',2:'#f58518',3:'#54a24b',4:'#e45756',5:'#b279a2'}
fig,ax=plt.subplots(2,2,figsize=(11,7),sharex=True)
for j,arm in enumerate(['smooth_joint','sequential']):
    for i,(k,t) in enumerate([('cifar100_test_top1','CIFAR-100 test top-1 (%)'),('tin200_val_top1','Tiny-ImageNet val top-1 (%)')]):
        a=ax[i][j]
        for s in range(1,6):
            r=R[(arm,s)]; a.plot([x['step'] for x in r],[x[k] for x in r],color=col[s],lw=1.6,label=f'seed {s}')
        if arm=='sequential': a.axvline(30000,color='gray',ls='--',lw=1); a.text(30500,a.get_ylim()[1]*0.5,'switch to target',fontsize=8,color='gray')
        a.set_title(f'ViT-Small {arm}'); a.set_ylabel(t); a.grid(alpha=.3)
        if i==1: a.set_xlabel('step')
ax[0][0].legend(fontsize=8,loc='lower right')
fig.suptitle('Exp 06 ViT-Small per-seed eval curves (EXPLORATORY)'); fig.tight_layout(); fig.savefig('perseed_curves_small.png',dpi=130)
fig,ax=plt.subplots(1,2,figsize=(11,3.8),sharey=True)
for j,arm in enumerate(['smooth_joint','sequential']):
    for s in range(1,6):
        r=[x for x in R[(arm,s)] if x['train_loss_src'] is not None and x['step']<=30000]
        ax[j].plot([x['step'] for x in r],[x['train_loss_src'] for x in r],color=col[s],lw=1.6,label=f'seed {s}')
    ax[j].set_title(f'{arm}: source train loss, steps 3k-30k'); ax[j].set_xlabel('step'); ax[j].grid(alpha=.3)
ax[0].set_ylabel('train_loss_src'); ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig('src_loss_by_seed.png',dpi=130)
