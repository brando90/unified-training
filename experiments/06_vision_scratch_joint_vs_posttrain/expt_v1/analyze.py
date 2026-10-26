"""Exp 06 analysis: cell status over the full denominator, per-arm summaries, paired-by-seed contrasts, H1/H2 verdicts.

Reads <runs>/<model>/<arm>/seed<k>/{final.json,curve.jsonl} for every row of cells.tsv and writes to --out:
  analysis.json, tables.md, curves_<model>.png (accuracy vs optimizer step, mean over seeds, one line per arm).

  python analyze.py --runs $EXP06_ROOT/runs --cells cells.tsv --out $EXP06_ROOT/analysis
Decision rules are the ones frozen in PROTOCOL.md; this script implements them, it does not choose them.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

ARMS = ('target_only', 'sequential', 'sequential_replay', 'fixed_joint', 'smooth_joint')
JOINT = ('fixed_joint', 'smooth_joint')
COLORS = dict(zip(ARMS, ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4')))  # fixed order, never cycled
T975 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.306, 9: 2.2622, 10: 2.2281}
T9875 = {1: 25.4517, 2: 6.2053, 3: 4.1765, 4: 3.4954, 5: 3.1634, 6: 2.9687, 7: 2.8412, 8: 2.7515, 9: 2.685, 10: 2.6338}
MARGIN = 1.0  # non-inferiority margin, percentage points of CIFAR-100 top-1 (PROTOCOL.md)


def t_interval(xs, table=T975):
    n = len(xs)
    if n == 0:
        return {'n': 0, 'mean': None, 'lo': None, 'hi': None, 'sd': None}
    m = sum(xs) / n
    if n < 2:
        return {'n': n, 'mean': m, 'lo': None, 'hi': None, 'sd': None}
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = table.get(n - 1, 1.96 if table is T975 else 2.2414) * sd / math.sqrt(n)
    return {'n': n, 'mean': m, 'lo': m - h, 'hi': m + h, 'sd': sd}


def load_cells(runs: Path, cells_tsv: Path):
    cells = []
    with open(cells_tsv) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            d = runs / row['model'] / row['arm'] / f"seed{row['seed']}"
            c = {'task': int(row['task']), 'model': row['model'], 'arm': row['arm'], 'seed': int(row['seed']),
                 'dir': str(d), 'status': 'unstarted'}
            fin = None
            if (d / 'final.json').exists():
                try:
                    fin = json.loads((d / 'final.json').read_text())
                    fin['final_cifar100_test_top1'], fin['final_tin200_val_top1']
                except Exception:
                    c['status'] = 'failed_unreadable_final'
            if fin is not None:
                c.update(status=fin['status'], cifar=fin['final_cifar100_test_top1'], tin=fin['final_tin200_val_top1'],
                         img_per_s=fin.get('img_per_s'), hours=(fin.get('train_seconds') or 0) / 3600,
                         peak_mem_gib=fin.get('peak_mem_gib'), skipped_frac=fin.get('skipped_frac'),
                         src_images=fin.get('src_images'), tgt_images=fin.get('tgt_images'))
            elif (d / 'ckpt.pt').exists():
                c['status'] = 'incomplete'
            if (d / 'curve.jsonl').exists():
                c['curve'] = [json.loads(l) for l in (d / 'curve.jsonl').read_text().splitlines() if l.strip()]
            cells.append(c)
    return cells


def paired(cells, model, a, b, key):
    by = {(c['arm'], c['seed']): c for c in cells if c['model'] == model and c['status'] == 'complete'}
    seeds = sorted({s for (arm, s) in by if arm == a} & {s for (arm, s) in by if arm == b})
    return seeds, [by[(a, s)][key] - by[(b, s)][key] for s in seeds]


def h1_verdict(ci, margin=MARGIN):
    if ci['lo'] is None:
        return 'not evaluable (<2 complete seed pairs)'
    if ci['lo'] > 0:
        return 'joint better'
    if ci['hi'] < 0:
        return 'sequential better'
    if ci['lo'] > -margin:
        return 'joint non-inferior'
    return 'inconclusive'


def analyze(cells, models):
    res = {'denominator': {}, 'arms': {}, 'contrasts': {}, 'verdicts': {}}
    for m in models:
        mc = [c for c in cells if c['model'] == m]
        res['denominator'][m] = {s: sum(c['status'] == s for c in mc) for s in sorted({c['status'] for c in mc})}
        res['denominator'][m]['total'] = len(mc)
        res['arms'][m] = {}
        for arm in ARMS:
            done = [c for c in mc if c['arm'] == arm and c['status'] == 'complete']
            res['arms'][m][arm] = {
                'complete': len(done), 'planned': sum(c['arm'] == arm for c in mc),
                'cifar': t_interval([c['cifar'] for c in done]), 'tin': t_interval([c['tin'] for c in done]),
                'img_per_s': t_interval([c['img_per_s'] for c in done if c.get('img_per_s')])['mean'],
                'hours': t_interval([c['hours'] for c in done])['mean']}
        means = {a: res['arms'][m][a]['cifar']['mean'] for a in JOINT}
        cand = [a for a in JOINT if means[a] is not None]
        best = max(cand, key=lambda a: means[a]) if cand else None
        con = {'best_joint': best}
        if best:
            for name, a, b, key, table in (
                    ('H1_best_joint_minus_sequential_cifar', best, 'sequential', 'cifar', T9875),
                    ('H2_best_joint_minus_sequential_tin', best, 'sequential', 'tin', T975),
                    ('best_joint_minus_sequential_replay_cifar', best, 'sequential_replay', 'cifar', T975),
                    ('best_joint_minus_sequential_replay_tin', best, 'sequential_replay', 'tin', T975),
                    ('best_joint_minus_target_only_cifar', best, 'target_only', 'cifar', T975),
                    ('sequential_minus_target_only_cifar', 'sequential', 'target_only', 'cifar', T975),
                    ('fixed_joint_minus_sequential_cifar', 'fixed_joint', 'sequential', 'cifar', T9875),
                    ('smooth_joint_minus_sequential_cifar', 'smooth_joint', 'sequential', 'cifar', T9875)):
                seeds, d = paired(cells, m, a, b, key)
                con[name] = {'a': a, 'b': b, 'metric': key, 'seeds': seeds, 'diffs': d,
                             'level': '97.5% (Bonferroni over 2 joint arms)' if table is T9875 else '95%',
                             **t_interval(d, table)}
        res['contrasts'][m] = con
        if best:
            h1 = con['H1_best_joint_minus_sequential_cifar']
            h2 = con['H2_best_joint_minus_sequential_tin']
            res['verdicts'][m] = {
                'H1': h1_verdict(h1), 'H1_supported': h1_verdict(h1) in ('joint better', 'joint non-inferior'),
                'H2': ('not evaluable (<2 complete seed pairs)' if h2['lo'] is None else
                       'supported' if h2['lo'] > 0 else 'not supported' if h2['hi'] < 0 else 'inconclusive')}
        else:
            res['verdicts'][m] = {'H1': 'not evaluable (no complete joint arm)', 'H1_supported': False,
                                  'H2': 'not evaluable (no complete joint arm)'}
    if 'tiny' in res['verdicts'] and 'small' in res['verdicts']:
        res['verdicts']['scale_check'] = ('same H1 verdict at ViT-Small' if res['verdicts']['small']['H1'] ==
                                          res['verdicts']['tiny']['H1'] else 'H1 verdict differs at ViT-Small')
    return res


def fmt(ci):
    if ci['mean'] is None:
        return 'n/a'
    if ci['lo'] is None:
        return f"{ci['mean']:.2f} (n={ci['n']})"
    return f"{ci['mean']:.2f} [{ci['lo']:.2f}, {ci['hi']:.2f}] (n={ci['n']})"


def tables_md(res, cells):
    out = ['# Exp 06 analysis tables', '',
           'Final-step top-1 accuracy (%), mean [95% t interval] over complete seeds. Contrasts are paired by seed;',
           'H1 contrasts use 97.5% intervals (Bonferroni over the two joint arms). p-val=n/a (intervals, no test).', '']
    for m in res['arms']:
        den = res['denominator'][m]
        out += [f'## ViT-{m.capitalize()}', '',
                'Cells: ' + ', '.join(f'{k} {v}' for k, v in den.items() if k != 'total') + f" of {den['total']}", '',
                '| Arm | Complete | CIFAR-100 test top-1 | Tiny-ImageNet val top-1 | img/s | train h |',
                '|---|---:|---|---|---:|---:|']
        for arm, r in res['arms'][m].items():
            ips = f"{r['img_per_s']:.0f}" if r['img_per_s'] else 'n/a'
            hrs = f"{r['hours']:.2f}" if r['hours'] is not None else 'n/a'
            out.append(f"| {arm} | {r['complete']}/{r['planned']} | {fmt(r['cifar'])} | {fmt(r['tin'])} | {ips} | {hrs} |")
        con = res['contrasts'][m]
        out += ['', f"Best joint arm (higher mean CIFAR-100): **{con['best_joint']}**", '',
                '| Contrast (a - b, pp) | Level | Seeds | Mean [interval] |', '|---|---|---|---|']
        for name, c in con.items():
            if name == 'best_joint':
                continue
            out.append(f"| {name} ({c['a']} - {c['b']}) | {c['level']} | {c['seeds']} | {fmt(c)} |")
        v = res['verdicts'][m]
        out += ['', f"**H1 (joint >= sequential on CIFAR-100):** {v['H1']}  ", f"**H2 (joint retains more Tiny-ImageNet):** {v['H2']}", '']
    if 'scale_check' in res['verdicts']:
        out += [f"**Scale check:** {res['verdicts']['scale_check']}", '']
    bad = [c for c in cells if c['status'] != 'complete']
    out += ['## Cells not complete', ''] + ([f"- task {c['task']}: {c['model']}/{c['arm']}/seed{c['seed']}: {c['status']}"
                                            for c in bad] or ['- none'])
    return '\n'.join(out) + '\n'


def plot(cells, model, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor='#fcfcfb')
    for ax, key, title in ((axes[0], 'cifar100_test_top1', 'CIFAR-100 test top-1 (target)'),
                           (axes[1], 'tin200_val_top1', 'Tiny-ImageNet val top-1 (source retention)')):
        for arm in ARMS:
            curves = [c['curve'] for c in cells if c['model'] == model and c['arm'] == arm and c['status'] == 'complete'
                      and c.get('curve')]
            if not curves:
                continue
            n = min(len(cv) for cv in curves)
            steps = [curves[0][i]['step'] for i in range(n)]
            mean = [sum(cv[i][key] for cv in curves) / len(curves) for i in range(n)]
            ax.plot(steps, mean, color=COLORS[arm], lw=2, label=f'{arm} (n={len(curves)})')
            ax.annotate(arm, (steps[-1], mean[-1]), xytext=(4, 0), textcoords='offset points', fontsize=7,
                        color='#52514e', va='center')
        ax.set_title(title, fontsize=10, color='#0b0b0b')
        ax.set_xlabel('optimizer step (matched across arms)', color='#52514e')
        ax.set_ylabel('top-1 %', color='#52514e')
        ax.grid(alpha=0.25)
        ax.set_facecolor('#fcfcfb')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f'Exp 06 ViT-{model.capitalize()}: mean over seeds', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--cells', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = load_cells(Path(a.runs), Path(a.cells))
    models = list(dict.fromkeys(c['model'] for c in cells))
    res = analyze(cells, models)
    res['cells'] = [{k: v for k, v in c.items() if k != 'curve'} for c in cells]
    (out / 'analysis.json').write_text(json.dumps(res, indent=1))
    md = tables_md(res, cells)
    (out / 'tables.md').write_text(md)
    for m in models:
        plot(cells, m, out / f'curves_{m}.png')
    print(md)


if __name__ == '__main__':
    main()
