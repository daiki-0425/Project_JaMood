"""Per-song CCC for JaMood held-out folds and RWC-P with each JaMood fold model."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mert_full_common import ROOT, FullMERTDataset, load_dataset, predict_labels
from train_mert_lstm import LSTM, collate_fn


def song_ccc(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if len(y) < 2:
        return float('nan')
    denominator = y.var() + p.var() + (y.mean() - p.mean()) ** 2
    if denominator == 0:
        return float('nan')
    return float(2 * np.mean((y - y.mean()) * (p - p.mean())) / denominator)


def infer(model, dataset, device, batch_size, description):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    predictions, targets = [], []
    with torch.inference_mode():
        for features, target, frames, mask, lengths in tqdm(loader, desc=description):
            output = predict_labels(model, features.to(device), frames.to(device), lengths).cpu()
            predictions.append(output[mask].numpy())
            targets.append(target[mask].numpy())
    p, y = np.concatenate(predictions), np.concatenate(targets)
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Non-finite predictions or targets')
    np.testing.assert_allclose(y, dataset.label_df[['valence', 'arousal']].to_numpy(), atol=1e-7)
    return dataset.label_df.assign(pred_valence=p[:, 0], pred_arousal=p[:, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models/jamood/mert/layer5/bilstm')
    parser.add_argument('--fold-labels', type=Path, default=ROOT / 'data/jamood/learning/mert_dynamic_vafolds.parquet')
    parser.add_argument('--folds', type=int, nargs='+', choices=range(5), default=list(range(5)))
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0 or len(set(args.folds)) != len(args.folds):
        parser.error('Use positive batch size and unique folds')
    jamood, jm_meta, _ = load_dataset('jamood')
    rwc, _, _ = load_dataset('rwc-p')
    folds = pd.read_parquet(args.fold_labels)
    folds['song_id'] = folds.song_id.astype(str)
    if folds.groupby('song_id').fold.nunique().ne(1).any() or not folds.fold.isin(range(5)).all():
        raise ValueError('Each JaMood song must have exactly one valid fold')
    keys = ['song_id', 'timestamp']
    if folds.duplicated(keys).any():
        raise ValueError('Duplicate training label keys')
    joined = jamood.label_df.merge(folds, on=keys, how='left', suffixes=('', '_training'), validate='one_to_one')
    if joined.fold.isna().any():
        raise ValueError('Missing fold assignments')
    for column in ['valence', 'arousal']:
        np.testing.assert_allclose(joined[column], joined[f'{column}_training'], atol=0, rtol=0)
    if not joined.mert_name.eq(joined.mert_name_training).all():
        raise ValueError('Feature names differ from training labels')
    for fold in args.folds:
        if not (args.model_dir / f'best_fold{fold}.pt').is_file():
            raise FileNotFoundError(args.model_dir / f'best_fold{fold}.pt')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = dict(status='running', model='JaMood MERT layer5 BiLSTM', folds=args.folds,
                    model_dir=str(args.model_dir.resolve()), fold_labels=str(args.fold_labels.resolve()),
                    device=args.device, batch_size=args.batch_size,
                    jamood_split='test fold = model fold; validation fold = (model fold + 1) % 5',
                    rwc_p_split='All 100 external songs for every model fold; RWC-P has no assigned test fold',
                    target_scale=[-1, 1], audio_offset_sec=14.5,
                    prediction_sampling='Nearest frame at label timestamp, without smoothing',
                    undefined_ccc='NaN if fewer than 2 labels or zero denominator; excluded from mean/median',
                    completed=[])
    def save_metadata():
        (args.output_dir / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    save_metadata()
    rows, summaries = [], []
    try:
        for fold in args.folds:
            model = LSTM().to(args.device)
            model.load_state_dict(torch.load(args.model_dir / f'best_fold{fold}.pt', map_location='cpu', weights_only=True), strict=True)
            model.eval()
            test = joined.loc[joined.fold == fold, jamood.label_df.columns].reset_index(drop=True)
            if test.empty:
                raise ValueError(f'Empty test fold {fold}')
            held_out = FullMERTDataset(test, jamood.mert_dir, jm_meta['audio_offset_sec'])
            for name, dataset in [('jamood', held_out), ('rwc-p', rwc)]:
                predictions = infer(model, dataset, args.device, args.batch_size, f'fold {fold} -> {name}')
                predictions.insert(0, 'model_fold', fold)
                predictions.insert(1, 'test_dataset', name)
                folder = args.output_dir / name / f'fold{fold}'
                folder.mkdir(parents=True)
                predictions.to_csv(folder / 'predictions.csv', index=False)
                pair_rows = []
                for song_id, song in predictions.groupby('song_id', sort=False):
                    row = dict(model_fold=fold, test_dataset=name, song_id=song_id,
                               n_labels=len(song), start_sec=song.timestamp.min()/1000,
                               end_sec=song.timestamp.max()/1000)
                    for label in ['valence', 'arousal']:
                        y, p = song[label].to_numpy(), song[f'pred_{label}'].to_numpy()
                        row[f'{label}_ccc'] = song_ccc(y, p)
                        row[f'{label}_rmse'] = float(np.sqrt(np.mean((y-p)**2)))
                        row[f'{label}_true_std'] = float(y.std())
                        row[f'{label}_pred_std'] = float(p.std())
                    pair_rows.append(row)
                per_song = pd.DataFrame(pair_rows)
                per_song.to_csv(folder / 'per_song_ccc.csv', index=False)
                rows.extend(pair_rows)
                summary = dict(model_fold=fold, test_dataset=name, n_songs=len(per_song), n_labels=len(predictions))
                for label in ['valence', 'arousal']:
                    values = per_song[f'{label}_ccc']
                    summary.update({f'{label}_song_ccc_mean': values.mean(),
                                    f'{label}_song_ccc_median': values.median(),
                                    f'{label}_song_ccc_std': values.std(),
                                    f'{label}_undefined_songs': int(values.isna().sum()),
                                    f'{label}_pooled_ccc': song_ccc(predictions[label], predictions[f'pred_{label}'])})
                summaries.append(summary)
                pd.DataFrame(rows).to_csv(args.output_dir / 'per_song_ccc.csv', index=False)
                pd.DataFrame(summaries).to_csv(args.output_dir / 'fold_summary.csv', index=False)
                metadata['completed'].append(dict(model_fold=fold, test_dataset=name))
                save_metadata()
                print(f'Saved fold {fold} -> {name}: {len(per_song)} songs', flush=True)
    except BaseException as exc:
        metadata.update(status='failed', error=str(exc))
        save_metadata()
        raise
    all_rows = pd.DataFrame(rows)
    for name in ['jamood', 'rwc-p']:
        all_rows[all_rows.test_dataset == name].to_csv(args.output_dir / f'{name}_per_song_ccc.csv', index=False)
    for label in ['valence', 'arousal']:
        all_rows[all_rows.test_dataset == 'rwc-p'].pivot(index='song_id', columns='model_fold', values=f'{label}_ccc').to_csv(
            args.output_dir / f'rwc-p_{label}_ccc_by_fold.csv')
    metadata['status'] = 'completed'
    save_metadata()
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == '__main__':
    main()
