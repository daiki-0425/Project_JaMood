"""Plot cross-dataset predictions in the existing RWC-P time-series style."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import quote

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NAMES = {'jamood': 'JaMood', 'deam': 'DEAM', 'memo2496': 'MEMO2496', 'pmemo': 'PMEmo'}
TARGETS = ['valence', 'arousal']


def configure_axis(ax, source, target, label):
    truth, = ax.plot([], [], color='#0072B2', linewidth=2, label=f'{NAMES[target]} ground truth')
    pred, = ax.plot([], [], color='#D55E00', linewidth=1.8, linestyle='--', label=f'{NAMES[source]} prediction')
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([-1, -.5, 0, .5, 1])
    ax.set_ylabel(label.capitalize())
    ax.set_xlabel('Time (s)')
    ax.grid(alpha=.2)
    ax.axhline(0, color='gray', linewidth=.7, alpha=.5)
    ax.legend(loc='upper right', fontsize=8)
    return truth, pred


def update_axis(ax, lines, song_id, song, label):
    time = song.timestamp.to_numpy() / 1000
    lines[0].set_data(time, song[label])
    lines[1].set_data(time, song[f'pred_{label}'])
    ax.set_xlim(time[0], time[-1] if time[-1] > time[0] else time[0] + 1)
    ax.set_title(f'Song {song_id} | {label.capitalize()}', fontsize=12)


def render_pair(input_root, output_root, source, target):
    input_dir = input_root / source / target
    df = pd.read_csv(input_dir / 'predictions.csv', dtype={'song_id': str})
    meta = json.loads((input_dir / 'metrics.json').read_text())
    columns = ['timestamp', 'valence', 'arousal', 'pred_valence', 'pred_arousal']
    if df.empty or df.song_id.isna().any() or not np.isfinite(df[columns].to_numpy()).all():
        raise ValueError(f'{source} -> {target}: empty/non-finite predictions')
    if df.duplicated(['song_id', 'timestamp']).any():
        raise ValueError('Duplicate song timestamps')
    if meta['train_dataset'] != source or meta['test_dataset'] != target:
        raise ValueError('Prediction metadata does not match selected pair')
    if len(df) != meta['n_labels'] or df.song_id.nunique() != meta['n_songs']:
        raise ValueError('Prediction counts do not match metadata')
    song_ids = sorted(df.song_id.unique(), key=lambda value: (0, int(value)) if value.isdigit() else (1, value))
    songs = {sid: group.sort_values('timestamp') for sid, group in df.groupby('song_id', sort=False)}
    out = output_root / f'{source}_to_{target}'
    out.mkdir(parents=True, exist_ok=False)
    preview_ids = song_ids[:4]
    preview, axes = plt.subplots(len(preview_ids), 2, figsize=(14, 11), squeeze=False)
    preview.suptitle(f'{NAMES[target]} ground truth vs. {NAMES[source]} predictions | First {len(preview_ids)} songs by ID', fontsize=16)
    for row, sid in enumerate(preview_ids):
        for col, label in enumerate(TARGETS):
            lines = configure_axis(axes[row, col], source, target, label)
            update_axis(axes[row, col], lines, sid, songs[sid], label)
    preview.tight_layout(rect=(0, 0, 1, .96))
    preview.savefig(out / 'preview.png', dpi=160)
    plt.close(preview)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5))
    fig.suptitle(f'{NAMES[source]}-trained MERT full model → {NAMES[target]}', fontsize=15)
    lines = [configure_axis(ax, source, target, label) for ax, label in zip(axes, TARGETS)]
    # Reuse axes and artists across songs; keep the layout identical in every image.
    fig.subplots_adjust(left=.07, right=.995, bottom=.075, top=.915, hspace=.3)
    index = []
    with PdfPages(out / 'all_songs.pdf') as pdf:
        for page, sid in enumerate(song_ids, 1):
            song = songs[sid]
            for ax, artists, label in zip(axes, lines, TARGETS):
                update_axis(ax, artists, sid, song, label)
            filename = f'song_{sid.zfill(4) if sid.isdigit() else quote(sid, safe="")}.png'
            fig.savefig(out / filename, dpi=150)
            pdf.savefig(fig)
            index.append(dict(song_id=sid, png=filename, pdf_page=page, n_labels=len(song),
                              start_sec=song.timestamp.iloc[0]/1000, end_sec=song.timestamp.iloc[-1]/1000))
            if page % 250 == 0:
                print(f'{source} -> {target}: {page}/{len(song_ids)} songs', flush=True)
    plt.close(fig)
    pd.DataFrame(index).to_csv(out / 'song_index.csv', index=False)
    (out / 'metadata.json').write_text(json.dumps(dict(
        source=str((input_dir / 'predictions.csv').resolve()), train_dataset=source,
        test_dataset=target, n_songs=len(song_ids), n_labels=len(df),
        preview_song_ids=preview_ids, smoothing=False,
        time_axis='Original annotation timestamps in seconds', y_limits=[-1.05, 1.05],
        prediction_sampling='Nearest MERT frame; no window averaging',
        predictions_outside_display_range=int((df[['pred_valence', 'pred_arousal']].abs() > 1.05).sum().sum()),
    ), indent=2) + '\n')
    print(f'Completed {source} -> {target}: {len(song_ids)} songs', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=ROOT / 'outputs/mert_cross_dataset/all_models_20261029')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'analysis/figures/mert_cross_dataset_all_models_20261029')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pair', nargs=2, choices=NAMES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error('--workers must be positive')
    if args.pair:
        render_pair(args.input_root, args.output_root, *args.pair)
        return
    run = json.loads((args.input_root / 'run.json').read_text())
    if run['status'] != 'completed' or run['completed_pairs'] != run['pairs']:
        raise ValueError('Evaluation must be complete before plotting all pairs')
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', pairs=run['pairs'], completed_pairs=[])
    status_path = args.output_root / 'plot_run.json'
    def save_status():
        status_path.write_text(json.dumps(manifest, indent=2) + '\n')
    def worker(pair):
        subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        '--input-root', str(args.input_root.resolve()),
                        '--output-root', str(args.output_root.resolve()), '--pair', *pair], check=True)
        return pair
    save_status()
    try:
        from concurrent.futures import as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(worker, pair) for pair in run['pairs']]
            for future in as_completed(futures):
                manifest['completed_pairs'].append(future.result())
                save_status()
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc))
        save_status()
        raise
    lines = ['# Cross-dataset prediction plots', '',
             'Blue solid: ground truth. Orange dashed: prediction. No smoothing or window averaging.', '',
             '| Train → Test | Preview | All songs PDF | Song index |', '|---|---|---|---|']
    for source, target in run['pairs']:
        folder = f'{source}_to_{target}'
        lines.append(f'| {NAMES[source]} → {NAMES[target]} | [PNG]({folder}/preview.png) | [PDF]({folder}/all_songs.pdf) | [CSV]({folder}/song_index.csv) |')
    (args.output_root / 'README.md').write_text('\n'.join(lines) + '\n')
    manifest['status'] = 'completed'
    save_status()
    print(f'Saved all plots: {args.output_root}', flush=True)


if __name__ == '__main__':
    main()
