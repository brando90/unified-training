"""Exp 06 cell: train one small ViT (plain torch) on Tiny-ImageNet-200 (source) + CIFAR-100 (target) under one arm.

Shared trunk, 200-way source head, 100-way target head. Every arm runs the same S optimizer steps at the same batch B
(matched FLOPs); every source-using arm sees exactly 0.5*S*B source images. Evaluates CIFAR-100 test top-1 (target
head) and Tiny-ImageNet val top-1 (source head) every --eval-every steps into curve.jsonl; checkpoints at the same
cadence and resumes from <out>/ckpt.pt when rerun with the same arguments.

  python train.py --arm fixed_joint --model tiny --seed 1 --data $EXP06_ROOT/data --out runs/tiny/fixed_joint/seed1
Exit codes: 0 complete, 2 failed (non-finite training), 3 stopped early (--stop-after or SIGUSR1/SIGTERM; rerun resumes).
"""
from __future__ import annotations
import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARMS = ('target_only', 'sequential', 'sequential_replay', 'fixed_joint', 'smooth_joint')
MODELS = {  # dim, depth, heads
    'tiny': (192, 12, 3),
    'small': (384, 12, 6),
    'smoke': (32, 2, 2),  # CPU smoke only
}
N_SRC_CLASSES, N_TGT_CLASSES = 200, 100


# ----------------------------------------------------------------------------------------------- model
class DropPath(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= self.p
        return x * keep.to(x.dtype) / (1.0 - self.p)


class Block(nn.Module):
    def __init__(self, dim, heads, drop_path):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.dp = DropPath(drop_path)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        x = x + self.dp(self.proj(a))
        return x + self.dp(self.fc2(F.gelu(self.fc1(self.n2(x)))))


class ViT(nn.Module):
    """ViT for 32x32 inputs, patch 4 (64 patches + CLS), linear patch embedding, pre-norm blocks, two heads."""

    def __init__(self, dim, depth, heads, patch=4, img=32, drop_path=0.1):
        super().__init__()
        self.patch = patch
        n = (img // patch) ** 2
        self.embed = nn.Linear(3 * patch * patch, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        self.blocks = nn.ModuleList(Block(dim, heads, drop_path * i / max(depth - 1, 1)) for i in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head_src = nn.Linear(dim, N_SRC_CLASSES)
        self.head_tgt = nn.Linear(dim, N_TGT_CLASSES)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def features(self, x):  # x: (B,3,32,32) float
        b, c, h, w = x.shape
        p = self.patch
        x = x.view(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // p) * (w // p), c * p * p)
        x = torch.cat([self.cls.expand(b, -1, -1), self.embed(x)], 1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x[:, 0])


def param_groups(model, wd):
    decay, no = [], []
    for n, p in model.named_parameters():
        (no if p.ndim < 2 or n in ('cls', 'pos') else decay).append(p)
    return [{'params': decay, 'weight_decay': wd}, {'params': no, 'weight_decay': 0.0}]


# ----------------------------------------------------------------------------------------------- schedule
def arm_plan(arm, S):
    """Return (source fraction per step f[0..S-1], optimizer phases [(start, end), ...])."""
    assert S % 10 == 0, 'S must be a multiple of 10 so every phase boundary is an integer step'
    t = np.arange(S, dtype=np.float64)
    if arm == 'target_only':
        return np.zeros(S), [(0, S)]
    if arm == 'sequential':
        return (t < S // 2).astype(np.float64), [(0, S // 2), (S // 2, S)]
    if arm == 'sequential_replay':
        b = 4 * S // 10
        return np.where(t < b, 1.0, 1.0 / 6.0), [(0, b), (b, S)]
    if arm == 'fixed_joint':
        return np.full(S, 0.5), [(0, S)]
    if arm == 'smooth_joint':
        return 1.0 - t / (S - 1), [(0, S)]
    raise ValueError(arm)


def source_counts(f, B):
    """Per-step source sample counts whose running total tracks B*cumsum(f) exactly (no drift from rounding)."""
    cum = np.round(np.concatenate([[0.0], np.cumsum(f)]) * B).astype(np.int64)
    n = np.diff(cum)
    assert n.min() >= 0 and n.max() <= B
    return n


def lr_at(step, phases, peak, warmup_frac):
    for s, e in phases:
        if s <= step < e:
            L, k = e - s, step - s
            W = max(1, round(warmup_frac * L))
            if k < W:
                return peak * (k + 1) / W
            return peak * 0.5 * (1.0 + math.cos(math.pi * (k - W) / max(1, L - W)))
    raise ValueError(step)


# ----------------------------------------------------------------------------------------------- data
class Stream:
    """Epoch-permutation sampler over a device-resident dataset; state (epoch, pos) is checkpointed."""

    def __init__(self, n, seed, ds_id, device):
        self.n, self.seed, self.ds_id, self.device = n, seed, ds_id, device
        self.epoch, self.pos = 0, 0
        self._perm()

    def _perm(self):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + self.ds_id * 7919 + self.epoch)
        self.perm = torch.randperm(self.n, generator=g).to(self.device)

    def take(self, k):
        out = []
        while k > 0:
            m = min(k, self.n - self.pos)
            out.append(self.perm[self.pos:self.pos + m])
            self.pos += m
            k -= m
            if self.pos == self.n:
                self.epoch, self.pos = self.epoch + 1, 0
                self._perm()
        return torch.cat(out) if out else torch.empty(0, dtype=torch.long, device=self.device)

    def state(self):
        return {'epoch': self.epoch, 'pos': self.pos}

    def load(self, s):
        self.epoch, self.pos = s['epoch'], s['pos']
        self._perm()


def augment(x_u8, gen):
    """GPU-side random crop (pad 4, reflect) + horizontal flip on a uint8 NHWC batch; returns normalized NCHW float."""
    x = x_u8.permute(0, 3, 1, 2).float().div_(255.0)
    b = x.shape[0]
    if b == 0:
        return x
    xp = F.pad(x, (4, 4, 4, 4), mode='reflect')
    oy = torch.randint(0, 9, (b,), device=x.device, generator=gen)
    ox = torch.randint(0, 9, (b,), device=x.device, generator=gen)
    ar = torch.arange(32, device=x.device)
    iy = (oy[:, None] + ar)[:, None, :, None].expand(b, 3, 32, 40)
    xp = xp.gather(2, iy)
    ix = (ox[:, None] + ar)[:, None, None, :].expand(b, 3, 32, 32)
    x = xp.gather(3, ix)
    flip = torch.rand(b, device=x.device, generator=gen) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    return (x - 0.5) / 0.5


def to_input(x_u8):
    return (x_u8.permute(0, 3, 1, 2).float().div(255.0) - 0.5) / 0.5


@torch.no_grad()
def evaluate(model, x, y, head, amp_ctx, bs=1000):
    model.eval()
    correct = 0
    for i in range(0, len(x), bs):
        with amp_ctx():
            logits = getattr(model, head)(model.features(to_input(x[i:i + bs])))
        correct += (logits.argmax(1) == y[i:i + bs]).sum().item()
    model.train()
    return 100.0 * correct / len(x)


# ----------------------------------------------------------------------------------------------- main
def atomic_save(obj, path):
    tmp = str(path) + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', required=True, choices=ARMS)
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=60000, help='S, total optimizer steps (all arms)')
    ap.add_argument('--batch', type=int, default=192, help='B, images per step (all arms); divisible by 6')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--warmup-frac', type=float, default=0.05, help='warmup fraction of each optimizer phase')
    ap.add_argument('--label-smoothing', type=float, default=0.1)
    ap.add_argument('--drop-path', type=float, default=0.1)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--eval-every', type=int, default=0, help='default S/20')
    ap.add_argument('--amp', default='bf16', choices=['bf16', 'fp16', 'none'], help='fp16 uses GradScaler (V100)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--stop-after', type=int, default=0, help='exit (code 3) after this many steps in this process')
    ap.add_argument('--max-bad-steps', type=int, default=50, help='consecutive non-finite losses that fail the cell')
    a = ap.parse_args()
    S, B = a.steps, a.batch
    assert B % 6 == 0, 'B must be divisible by 6 (exact 1/6 replay) and 2 (exact 50/50)'
    eval_every = a.eval_every or S // 20
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    final_path, ckpt_path, curve_path = out / 'final.json', out / 'ckpt.pt', out / 'curve.jsonl'
    cfg = {k: v for k, v in vars(a).items() if k not in ('out', 'data', 'device', 'stop_after')}
    cfg['eval_every'] = eval_every
    if final_path.exists() and json.loads(final_path.read_text()).get('status') == 'complete':
        print(f'{out}: already complete', flush=True)
        return 0

    dev = torch.device(a.device)
    if dev.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(a.amp)

    def amp_ctx():
        if amp_dtype is None:
            return torch.autocast(dev.type, enabled=False)
        return torch.autocast(dev.type, dtype=amp_dtype)

    d = Path(a.data)
    ld = lambda n: torch.from_numpy(np.load(d / n)).to(dev)
    src_x, src_y = ld('tin200_train_x.npy'), ld('tin200_train_y.npy')
    tgt_x, tgt_y = ld('cifar100_train_x.npy'), ld('cifar100_train_y.npy')
    ev_tgt_x, ev_tgt_y = ld('cifar100_test_x.npy'), ld('cifar100_test_y.npy')
    ev_src_x, ev_src_y = ld('tin200_val_x.npy'), ld('tin200_val_y.npy')

    f, phases = arm_plan(a.arm, S)
    n_src = source_counts(f, B)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dim, depth, heads = MODELS[a.model]
    model = ViT(dim, depth, heads, drop_path=a.drop_path).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    make_opt = lambda: torch.optim.AdamW(param_groups(model, a.wd), lr=a.lr, betas=(0.9, 0.999), eps=1e-8)
    opt = make_opt()
    scaler = torch.amp.GradScaler(dev.type, enabled=(a.amp == 'fp16'))
    src_stream, tgt_stream = Stream(len(src_x), a.seed, 1, dev), Stream(len(tgt_x), a.seed, 2, dev)
    step, src_seen, tgt_seen, skipped, bad_run, train_seconds, curve_lines = 0, 0, 0, 0, 0, 0.0, 0
    loss_sum = {'src': 0.0, 'tgt': 0.0, 'n_src': 0, 'n_tgt': 0}  # train-loss sums since the last eval

    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)  # RNG states must stay CPU tensors
        assert ck['cfg'] == cfg, f'checkpoint config differs:\n{ck["cfg"]}\n{cfg}'
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        scaler.load_state_dict(ck['scaler'])
        src_stream.load(ck['src_stream'])
        tgt_stream.load(ck['tgt_stream'])
        step, src_seen, tgt_seen = ck['step'], ck['src_seen'], ck['tgt_seen']
        skipped, bad_run, train_seconds, curve_lines = ck['skipped'], ck['bad_run'], ck['train_seconds'], ck['curve_lines']
        loss_sum.update(ck['loss_sum'])
        torch.set_rng_state(ck['rng_cpu'])
        if dev.type == 'cuda' and ck.get('rng_cuda') is not None:
            torch.cuda.set_rng_state(ck['rng_cuda'])
        lines = curve_path.read_text().splitlines()[:curve_lines] if curve_path.exists() else []
        curve_path.write_text(''.join(l + '\n' for l in lines))  # drop evals newer than the checkpoint
        print(f'resumed {out} at step {step}', flush=True)
    else:
        curve_path.write_text('')

    stop = {'flag': False}
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__('flag', True))

    def save_ckpt():
        atomic_save({'cfg': cfg, 'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                     'src_stream': src_stream.state(), 'tgt_stream': tgt_stream.state(), 'step': step,
                     'src_seen': src_seen, 'tgt_seen': tgt_seen, 'skipped': skipped, 'bad_run': bad_run,
                     'train_seconds': train_seconds, 'curve_lines': curve_lines, 'loss_sum': dict(loss_sum), 'rng_cpu': torch.get_rng_state(),
                     'rng_cuda': torch.cuda.get_rng_state() if dev.type == 'cuda' else None}, ckpt_path)

    def log_eval():
        nonlocal curve_lines
        rec = {'step': step, 'images': step * B, 'src_images': src_seen, 'tgt_images': tgt_seen,
               'cifar100_test_top1': evaluate(model, ev_tgt_x, ev_tgt_y, 'head_tgt', amp_ctx),
               'tin200_val_top1': evaluate(model, ev_src_x, ev_src_y, 'head_src', amp_ctx),
               'train_loss_src': loss_sum['src'] / loss_sum['n_src'] if loss_sum['n_src'] else None,
               'train_loss_tgt': loss_sum['tgt'] / loss_sum['n_tgt'] if loss_sum['n_tgt'] else None,
               'lr': lr_at(min(step, S - 1), phases, a.lr, a.warmup_frac), 'skipped_steps': skipped,
               'train_seconds': round(train_seconds, 1),
               'img_per_s': round(step * B / train_seconds, 1) if train_seconds else None}
        loss_sum.update(src=0.0, tgt=0.0, n_src=0, n_tgt=0)
        with open(curve_path, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        curve_lines += 1
        print(json.dumps(rec), flush=True)
        return rec

    if step == 0:
        log_eval()
        save_ckpt()
    phase_starts = {s for s, _ in phases[1:]}
    ran, status = 0, 'complete'
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t_last = time.time()
    while step < S:
        if step in phase_starts:
            # Pretrain -> finetune boundary: fresh AdamW state, LR warmup+cosine restarts (lr_at handles the LR).
            # A resume exactly at the boundary reaches here too and resets again, which is the same state.
            opt = make_opt()
        lr = lr_at(step, phases, a.lr, a.warmup_frac)
        for g in opt.param_groups:
            g['lr'] = lr
        k = int(n_src[step])
        gen = torch.Generator(device=dev).manual_seed(a.seed * 10_000_019 + step)
        si, ti = src_stream.take(k), tgt_stream.take(B - k)
        x = torch.cat([augment(src_x[si], gen), augment(tgt_x[ti], gen)])
        with amp_ctx():
            feats = model.features(x)
            ls = F.cross_entropy(model.head_src(feats[:k]), src_y[si], label_smoothing=a.label_smoothing,
                                 reduction='sum') if k else feats.new_zeros(())
            lt = F.cross_entropy(model.head_tgt(feats[k:]), tgt_y[ti], label_smoothing=a.label_smoothing,
                                 reduction='sum') if k < B else feats.new_zeros(())
            loss = (ls + lt) / B
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            skipped += 1
            bad_run += 1
            if bad_run >= a.max_bad_steps:
                status = 'failed_nonfinite'
                break
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            if not scaler.is_enabled() and not torch.isfinite(gnorm):
                skipped += 1  # bf16/fp32: finite loss but non-finite gradients; skip rather than poison the weights
                bad_run += 1
                if bad_run >= a.max_bad_steps:
                    status = 'failed_nonfinite'
                    break
                opt.zero_grad(set_to_none=True)
            else:
                bad_run = 0
                scale_before = scaler.get_scale() if scaler.is_enabled() else None
                scaler.step(opt)
                scaler.update()
                if scaler.is_enabled() and scaler.get_scale() < scale_before:
                    skipped += 1  # fp16 overflow: optimizer step skipped by GradScaler
            loss_sum['src'] += ls.item() if k else 0.0
            loss_sum['tgt'] += lt.item() if k < B else 0.0
            loss_sum['n_src'] += k
            loss_sum['n_tgt'] += B - k
        src_seen += k
        tgt_seen += B - k
        step += 1
        ran += 1
        if step % eval_every == 0 or step == S:
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            train_seconds += time.time() - t_last
            log_eval()
            save_ckpt()
            t_last = time.time()
        if step < S and ((a.stop_after and ran >= a.stop_after) or stop['flag']):
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            train_seconds += time.time() - t_last
            save_ckpt()
            print(f'stopped early at step {step}; rerun resumes', flush=True)
            return 3

    peak = torch.cuda.max_memory_allocated() / 2**30 if dev.type == 'cuda' else None
    curve = [json.loads(l) for l in curve_path.read_text().splitlines()]
    last = curve[-1]
    fin = {'status': status if status != 'complete' or last['step'] == S else 'incomplete', 'cfg': cfg,
           'arm': a.arm, 'model': a.model, 'seed': a.seed, 'n_params': n_params, 'steps_done': step,
           'src_images': src_seen, 'tgt_images': tgt_seen, 'skipped_steps': skipped,
           'skipped_frac': skipped / max(step, 1),
           'final_cifar100_test_top1': last['cifar100_test_top1'], 'final_tin200_val_top1': last['tin200_val_top1'],
           'final_step': last['step'], 'train_seconds': round(train_seconds, 1),
           'img_per_s': round(step * B / train_seconds, 1) if train_seconds else None, 'peak_mem_gib': peak,
           'device': torch.cuda.get_device_name() if dev.type == 'cuda' else 'cpu', 'torch': torch.__version__}
    if fin['status'] == 'complete' and fin['skipped_frac'] > 0.01:
        fin['status'] = 'failed_skipped_steps'
    final_path.write_text(json.dumps(fin, indent=1))
    print('FINAL ' + json.dumps({k: fin[k] for k in ('status', 'final_cifar100_test_top1', 'final_tin200_val_top1',
                                                     'src_images', 'tgt_images', 'img_per_s', 'peak_mem_gib')}),
          flush=True)
    return 0 if fin['status'] == 'complete' else 2


if __name__ == '__main__':
    sys.exit(main())
