"""Redraw curves_{tiny,small}.png from the raw runs with the legend only (no overlapping end labels).

Plot-only; reads the same curve.jsonl files as expt_v1/analyze.py and changes no number.
Usage: python replot.py --runs <exp06 runs dir> --out <this analysis dir>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'expt_v1'))
from analyze import ARMS, COLORS  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def load(runs, model, arm):
    out = []
    for d in sorted((runs / model / arm).glob('seed*')):
        if (d / 'final.json').exists() and (d / 'curve.jsonl').exists():
            out.append([json.loads(l) for l in (d / 'curve.jsonl').read_text().splitlines() if l.strip()])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    for model in ('tiny', 'small'):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor='#fcfcfb')
        for ax, key, title in ((axes[0], 'cifar100_test_top1', 'CIFAR-100 test top-1 (target)'),
                               (axes[1], 'tin200_val_top1', 'Tiny-ImageNet val top-1 (source retention)')):
            for arm in ARMS:
                curves = load(Path(a.runs), model, arm)
                if not curves:
                    continue
                n = min(len(c) for c in curves)
                steps = [curves[0][i]['step'] for i in range(n)]
                mean = [sum(c[i][key] for c in curves) / len(curves) for i in range(n)]
                ax.plot(steps, mean, color=COLORS[arm], lw=2, label=f'{arm} (n={len(curves)})')
            ax.set_title(title, fontsize=10, color='#0b0b0b')
            ax.set_xlabel('optimizer step (matched across arms)', color='#52514e')
            ax.set_ylabel('top-1 %', color='#52514e')
            ax.grid(alpha=0.25)
            ax.set_facecolor('#fcfcfb')
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
        axes[0].legend(fontsize=7, frameon=False, loc='center right', bbox_to_anchor=(1.0, 0.45))
        fig.suptitle(f'Exp 06 ViT-{model.capitalize()}: mean over seeds', fontsize=11)
        fig.tight_layout()
        fig.savefig(Path(a.out) / f'curves_{model}.png', dpi=130)
        plt.close(fig)


if __name__ == '__main__':
    main()
