"""保存済み MERT + BiLSTM を、学習時の fold 分割で評価する。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mert_dataset_lstm import MERTDataset
from train_mert_lstm import LSTM, collate_fn, ccc_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['pmemo', 'deam', 'memo2496'], required=True)
    parser.add_argument('--labels', type=Path, required=True,
                        help='学習時と同じラベル・スケール・fold の parquet')
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--feature-dir', type=Path)
    parser.add_argument('--audio-offset-sec', type=float, default=15.0)
    parser.add_argument('--folds', type=int, nargs='+', choices=range(5), default=list(range(5)))
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    model_dir = args.model_dir or Path(f'models/{args.dataset}/mert/layer5/bilstm')
    feature_dir = args.feature_dir or Path(f'data/{args.dataset}/feature/mert/layer5')
    df = pd.read_parquet(args.labels)
    if df.groupby('song_id')['fold'].nunique().gt(1).any():
        raise ValueError('同じ曲が複数の fold に含まれています')
    for fold in args.folds:
        if not (model_dir / f'best_fold{fold}.pt').is_file():
            raise FileNotFoundError(model_dir / f'best_fold{fold}.pt')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in args.folds:
        test_df = df[df['fold'] == fold]
        train_df = df[~df['fold'].isin([fold, (fold + 1) % 5])]
        if test_df.empty or train_df.empty:
            raise ValueError(f'fold {fold}: テストまたは学習データが空です')
        dataset = MERTDataset(test_df, feature_dir, args.audio_offset_sec)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn)
        model = LSTM().to(args.device)
        state = torch.load(model_dir / f'best_fold{fold}.pt',
                           map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        predictions, targets = [], []
        with torch.inference_mode():
            for features, target, frames, mask, lengths in tqdm(loader, desc=f'fold {fold} test'):
                output = model(features.to(args.device), lengths).cpu()
                picked = output[torch.arange(output.size(0)).unsqueeze(1), frames]
                predictions.append(picked[mask])
                targets.append(target[mask])
        y_pred = torch.cat(predictions).numpy().astype(np.float64)
        y_true = torch.cat(targets).numpy().astype(np.float64)
        if not np.isfinite(y_pred).all() or not np.isfinite(y_true).all():
            raise ValueError(f'fold {fold}: 予測またはラベルに NaN/Inf があります')
        row = {'fold': fold, 'n_songs': len(dataset), 'n_labels': len(y_true),
               'n_excluded_labels': len(test_df) - len(y_true)}
        for j, name in enumerate(['valence', 'arousal']):
            mse = np.mean((y_true[:, j] - y_pred[:, j]) ** 2)
            row.update({
                f'{name}_mse': mse,
                f'{name}_rmse': np.sqrt(mse),
                f'{name}_ccc': ccc_score(y_true[:, j], y_pred[:, j]),
                f'{name}_r2': r2_score(y_true[:, j], y_pred[:, j]),
                f'{name}_baseline_rmse': np.sqrt(np.mean(
                    (y_true[:, j] - train_df[name].mean()) ** 2)),
            })
        rows.append(row)
        print(pd.DataFrame([row]).to_string(index=False), flush=True)
        pd.DataFrame(rows).to_csv(args.output_dir / 'metrics.csv', index=False)
    metrics = pd.DataFrame(rows)
    columns = [c for c in metrics if c.startswith(('valence_', 'arousal_'))]
    # 各 fold を等しく重み付けした平均（全ラベルを結合した指標ではない）。
    summary = metrics[columns].agg(['mean', 'std'])
    summary.to_csv(args.output_dir / 'summary.csv', index_label='statistic')
    print('\nFold mean / std:\n' + summary.to_string())
    print(f'保存先: {args.output_dir}')


if __name__ == '__main__':
    main()
