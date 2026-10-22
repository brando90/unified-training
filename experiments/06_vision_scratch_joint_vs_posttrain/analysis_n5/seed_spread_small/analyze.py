import json,glob,statistics as S
import numpy as np
R={}
for f in sorted(glob.glob('raw/runs/small/*/seed*/curve.jsonl')):
    p=f.split('/'); R[(p[3],int(p[4][4:]))]=[json.loads(l) for l in open(f)]
arms=['target_only','sequential','sequential_replay','fixed_joint','smooth_joint']
def g(a,s,st,k):
    for r in R[(a,s)]:
        if r['step']==st: return r[k]
steps=[r['step'] for r in R[('smooth_joint',1)]]
print('sd of CIFAR-100 across 5 seeds, by step')
print('step',*arms)
for st in steps[1:]:
    print(st,*[round(S.stdev([g(a,s,st,'cifar100_test_top1') for s in range(1,6)]),2) for a in arms])
print('sd TIN by step')
for st in steps[1:]:
    print(st,*[round(S.stdev([g(a,s,st,'tin200_val_top1') for s in range(1,6)]),2) for a in arms])
# correlation of early values with final for smooth_joint
fin=[g('smooth_joint',s,60000,'cifar100_test_top1') for s in range(1,6)]
print('final',fin)
for st in [3000,6000,9000,12000,15000,18000,21000,30000]:
    c=[g('smooth_joint',s,st,'cifar100_test_top1') for s in range(1,6)]
    lt=[g('smooth_joint',s,st,'train_loss_tgt') for s in range(1,6)]
    ls=[g('smooth_joint',s,st,'train_loss_src') for s in range(1,6)]
    print(st,'cifar',c,'r(final)=%.2f'%np.corrcoef(c,fin)[0,1],'| loss_tgt',[round(x,2) for x in lt],'r=%.2f'%np.corrcoef(lt,fin)[0,1],'| loss_src',[round(x,2) for x in ls],'r=%.2f'%np.corrcoef(ls,fin)[0,1])
# per-cell: other arms' seed rank vs smooth_joint rank at 60000
for a in ['fixed_joint','sequential_replay','target_only','sequential']:
    o=[g(a,s,60000,'cifar100_test_top1') for s in range(1,6)]
    print(a,'final',o,'r with smooth_joint=%.2f'%np.corrcoef(o,fin)[0,1])
# smooth_joint src/tgt fraction by step (seed1)
print([ (r['step'],round(r['tgt_images']/max(1,r['images']),3)) for r in R[('smooth_joint',1)]][::2])
# lr
print([ (r['step'],round(r['lr'],5)) for r in R[('smooth_joint',1)]][::3])
print('--- source-loss by seed at step 21000 and 30000 (does the slow-seed pattern carry across arms?)')
for a in arms:
    for st in [12000,21000,30000]:
        ls=[g(a,s,st,'train_loss_src') for s in range(1,6)]
        if None in ls: print(a,st,'n/a'); continue
        print(a,st,[round(x,2) for x in ls])
print('--- loss_tgt at 60000 smooth_joint', [round(g('smooth_joint',s,60000,'train_loss_tgt'),3) for s in range(1,6)])
print('--- final src loss by arm-seed:')
for a in arms: print(a,[g(a,s,60000,'train_loss_src') and round(g(a,s,60000,'train_loss_src'),3) for s in range(1,6)])
