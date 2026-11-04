"""Evaluate a full-data MERT BiLSTM checkpoint on a separate dataset."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from train_mert_lstm import LSTM, collate_fn, ccc_score
from mert_full_common import DATASETS, dataset_name, load_dataset, predict_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, default='rwc-p')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--feature-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('batch-size must be positive')
    name = dataset_name(args.dataset)
    dataset, metadata, _ = load_dataset(name, args.labels, args.feature_dir)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['config']['target_scale'] != metadata['target_scale']:
        raise ValueError('Model and test label scales differ')
    if checkpoint['epoch'] != checkpoint['config']['epochs']:
        raise ValueError('Use the completed final.pt checkpoint')
    model = LSTM().to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    predictions, targets = [], []
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    with torch.inference_mode():
        for features, target, frames, mask, lengths in tqdm(loader, desc=f'test {name}'):
            picked = predict_labels(model, features.to(args.device), frames.to(args.device), lengths).cpu()
            predictions.append(picked[mask].numpy())
            targets.append(target[mask].numpy())
    p, y = np.concatenate(predictions).astype(float), np.concatenate(targets).astype(float)
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Non-finite predictions/targets')
    assert len(y) == len(dataset.label_df)
    np.testing.assert_allclose(y, dataset.label_df[['valence', 'arousal']].to_numpy(), atol=1e-7)
    rows = []
    for j, label in enumerate(['valence', 'arousal']):
        rows.append(dict(target=label, rmse=float(np.sqrt(np.mean((y[:, j] - p[:, j]) ** 2))),
                         r2=float(r2_score(y[:, j], p[:, j])), ccc=float(ccc_score(y[:, j], p[:, j]))))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / 'metrics.csv', index=False)
    dataset.label_df.assign(pred_valence=p[:, 0], pred_arousal=p[:, 1]).to_csv(args.output_dir / 'predictions.csv', index=False)
    summary = dict(train_dataset=checkpoint['config']['dataset'], test_dataset=name,
                   checkpoint=str(args.checkpoint.resolve()), epoch=checkpoint['epoch'],
                   target_scale=[-1, 1], n_songs=len(dataset), n_labels=len(y),
                   excluded_labels=metadata['excluded_labels'], metrics=rows)
    (args.output_dir / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(pd.DataFrame(rows).to_string(index=False))
    print(f'Saved: {args.output_dir}')


if __name__ == '__main__':
    main()
