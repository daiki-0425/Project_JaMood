"""Overlay true and predicted VA distributions in the existing RWC-P figure style."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NAMES = {'jamood': 'JaMood', 'deam': 'DEAM', 'memo2496': 'MEMO2496', 'pmemo': 'PMEmo'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=ROOT / 'outputs/mert_cross_dataset/all_models_20261029')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'analysis/figures/va_plot_per_datasets')
    args = parser.parse_args()
    run = json.loads((args.input_root / 'run.json').read_text())
    if run['status'] != 'completed':
        raise ValueError('Complete cross-dataset evaluation before plotting')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stems = [f'{source}_mert_full_to_{target}_va_overlay' for source, target in run['pairs']]
    outputs = [args.output_dir / f'{stem}.{ext}' for stem in stems for ext in ['png', 'pdf']]
    bundle = args.output_dir / 'cross_dataset_va_overlays_all.pdf'
    statistics = args.output_dir / 'cross_dataset_va_overlay_statistics.csv'
    readme = args.output_dir / 'cross_dataset_va_overlays.md'
    for path in outputs + [bundle, statistics, readme]:
        if path.exists():
            raise FileExistsError(f'Use a new --output-dir to avoid overwriting {path}')
    rows = []
    links = ['# Cross-dataset VA overlays', '',
             'Blue: ground truth. Orange: prediction. X markers: means over all valid timestamps.',
             'All points are plotted without sampling, smoothing, or window averaging. Axes: [-1, 1].', '',
             '| Train → Test | PNG | PDF |', '|---|---|---|']
    with PdfPages(bundle) as pdf:
        for (source, target), stem in zip(run['pairs'], stems):
            input_dir = args.input_root / source / target
            df = pd.read_csv(input_dir / 'predictions.csv', dtype={'song_id': str})
            meta = json.loads((input_dir / 'metrics.json').read_text())
            truth = df[['valence', 'arousal']].to_numpy()
            pred = df[['pred_valence', 'pred_arousal']].to_numpy()
            if len(df) != meta['n_labels'] or df.song_id.nunique() != meta['n_songs']:
                raise ValueError(f'{source} -> {target}: inconsistent counts')
            if meta['train_dataset'] != source or meta['test_dataset'] != target:
                raise ValueError('Dataset mismatch')
            if not np.isfinite(truth).all() or not np.isfinite(pred).all():
                raise ValueError('Non-finite points')
            means = [truth.mean(axis=0), pred.mean(axis=0)]
            fig, ax = plt.subplots(figsize=(9, 9))
            for values, color, label, mean in zip(
                    [truth, pred], ['#0072B2', '#D55E00'], ['Ground truth', 'Prediction'], means):
                ax.scatter(values[:, 0], values[:, 1], s=12, c=color, alpha=.7,
                           edgecolors='white', linewidths=.25, rasterized=True)
                ax.scatter(*mean, s=1100, marker='X', c=color, edgecolors='black', linewidths=2.5,
                           zorder=5, label=f'{label} mean ({mean[0]:.3f}, {mean[1]:.3f})')
            ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect='equal')
            ax.set_xticks(np.arange(-1, 1.01, .5))
            ax.set_yticks(np.arange(-1, 1.01, .5))
            ax.tick_params(labelsize=26)
            ax.set_xlabel('Valence', fontsize=36, labelpad=12)
            ax.set_ylabel('Arousal', fontsize=36, labelpad=12)
            ax.set_axisbelow(True)
            ax.grid(alpha=.15)
            ax.axhline(0, color='gray', linewidth=.8)
            ax.axvline(0, color='gray', linewidth=.8)
            ax.legend(loc='lower left', frameon=False, fontsize=20, labelspacing=.8)
            fig.tight_layout()
            fig.savefig(args.output_dir / f'{stem}.png', dpi=300, bbox_inches='tight')
            fig.savefig(args.output_dir / f'{stem}.pdf', dpi=300, bbox_inches='tight')
            ax.set_title(f'{NAMES[source]} → {NAMES[target]}', fontsize=20, pad=14)
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
            plt.close(fig)
            rows.append(dict(train_dataset=source, test_dataset=target, n_songs=meta['n_songs'],
                             n_labels=len(df), true_mean_valence=means[0][0], true_mean_arousal=means[0][1],
                             pred_mean_valence=means[1][0], pred_mean_arousal=means[1][1],
                             n_pred_points_outside_axes=int((np.abs(pred) > 1).any(axis=1).sum()),
                             predictions=str((input_dir / 'predictions.csv').resolve())))
            links.append(f'| {NAMES[source]} → {NAMES[target]} | [PNG]({stem}.png) | [PDF]({stem}.pdf) |')
            print(f'Saved {source} -> {target}: {len(df):,} points per distribution', flush=True)
    pd.DataFrame(rows).to_csv(statistics, index=False)
    readme.write_text('\n'.join(links) + '\n')
    print(f'Completed: {bundle}', flush=True)


if __name__ == '__main__':
    main()
