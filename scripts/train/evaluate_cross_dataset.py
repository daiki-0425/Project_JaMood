"""Cross-dataset evaluation of full-data MERT BiLSTM models."""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from mert_full_common import ROOT, dataset_name, load_dataset

DATASETS = ['jamood', 'deam', 'memo2496', 'pmemo']
DEFAULT_CHECKPOINTS = {
    name: ROOT / f"models/{name}/mert/layer5/bilstm/{run}/final.pt"
    for name, run in [('jamood', 'full_10epochs'), ('deam', 'full_10epochs'),
                      ('memo2496', 'full_20261025_052316'), ('pmemo', 'full_10epochs')]
}


def main(legacy_output=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-datasets', nargs='+', type=dataset_name,
                        choices=DATASETS + ['all'], default=['jamood'],
                        help='Training datasets, or all for all 12 directed pairs; memo aliases memo2496')
    parser.add_argument('--checkpoint', type=Path,
                        help='Override checkpoint for a single training dataset')
    parser.add_argument('--datasets', nargs='+', type=dataset_name, choices=DATASETS,
                        help='Test datasets; defaults to all except the training dataset')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data',
                        help='Root containing <dataset>/learning and <dataset>/feature')
    parser.add_argument('--output-dir', type=Path,
                        help='New directory; defaults to a timestamped outputs/mert_cross_dataset directory')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    if 'all' in args.train_datasets:
        if args.train_datasets != ['all']:
            parser.error('Use --train-datasets all by itself')
        args.train_datasets = DATASETS.copy()
    if len(set(args.train_datasets)) != len(args.train_datasets):
        parser.error('--train-datasets must not contain duplicates')
    if args.datasets and len(set(args.datasets)) != len(args.datasets):
        parser.error('--datasets must not contain duplicates')
    if args.checkpoint and len(args.train_datasets) != 1:
        parser.error('--checkpoint requires exactly one training dataset')
    if legacy_output and args.train_datasets != ['jamood']:
        parser.error('Use evaluate_cross_dataset.py for other training datasets')

    pairs = [(source, target) for source in args.train_datasets
             for target in (args.datasets or DATASETS) if source != target]
    if not pairs:
        parser.error('No cross-dataset pairs selected')
    checkpoints = {source: args.checkpoint or DEFAULT_CHECKPOINTS[source]
                   for source in args.train_datasets}
    # Validate all checkpoints and target inputs before starting an expensive run.
    for source, path in checkpoints.items():
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        config = checkpoint['config']
        if dataset_name(config['dataset']) != source:
            raise ValueError(f'{path}: expected a model trained on {source}')
        if config['target_scale'] != [-1, 1]:
            raise ValueError(f'{path}: expected target scale [-1, 1]')
        if checkpoint['epoch'] != config['epochs']:
            raise ValueError(f'{path}: use the completed final.pt checkpoint')
        del checkpoint
    inputs = {}
    for name in dict.fromkeys(target for _, target in pairs):
        labels = args.data_root / name / 'learning/mert_dynamic_va.parquet'
        features = args.data_root / name / 'feature/mert/layer5'
        load_dataset(name, labels, features)
        inputs[name] = (labels, features)

    output_dir = args.output_dir or (
        ROOT / 'outputs/mert_cross_dataset' / datetime.now().strftime('cross_%Y%m%d_%H%M%S_%f'))
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(train_datasets=args.train_datasets, pairs=pairs,
                    checkpoints={name: str(path.resolve()) for name, path in checkpoints.items()},
                    data_root=str(args.data_root.resolve()),
                    batch_size=args.batch_size, device=args.device, target_scale=[-1, 1],
                    aggregation='All valid annotation timestamps pooled within each dataset',
                    prediction_sampling='Nearest MERT frame at each label timestamp; no window averaging',
                    status='running', completed_pairs=[])
    manifest_path = output_dir / 'run.json'

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

    save_manifest()
    rows = []
    try:
        for source, name in pairs:
            labels, features = inputs[name]
            pair_dir = output_dir / name if legacy_output else output_dir / source / name
            print(f'\n{source} -> {name}', flush=True)
            subprocess.run([
                sys.executable, str(Path(__file__).with_name('test_mert_lstm_full.py')),
                '--dataset', name, '--checkpoint', str(checkpoints[source]),
                '--labels', str(labels), '--feature-dir', str(features),
                '--output-dir', str(pair_dir),
                '--batch-size', str(args.batch_size), '--device', args.device,
            ], check=True)
            result = json.loads((pair_dir / 'metrics.json').read_text())
            for metric in result['metrics']:
                rows.append(dict(train_dataset=source, test_dataset=name,
                                 n_songs=result['n_songs'], n_labels=result['n_labels'],
                                 excluded_labels=result['excluded_labels'], **metric))
            pd.DataFrame(rows).to_csv(output_dir / 'summary.csv', index=False)
            manifest['completed_pairs'].append([source, name])
            save_manifest()
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc))
        save_manifest()
        raise
    manifest['status'] = 'completed'
    save_manifest()
    print('\n' + pd.DataFrame(rows).to_string(index=False))
    print(f'Saved: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
