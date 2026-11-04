"""Shared input validation for fold-free MERT training and external evaluation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mert_dataset_lstm import MERTDataset

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ['jamood', 'pmemo', 'deam', 'memo2496', 'memo', 'rwc-p']


def dataset_name(value):
    return 'memo2496' if value == 'memo' else value


class FullMERTDataset(MERTDataset):
    def __getitem__(self, index):
        song_id = self.song_ids[index]
        try:
            features, targets, frames = super().__getitem__(index)
        except Exception as exc:
            filename = self.label_df.loc[self.label_df.song_id == song_id, 'mert_name'].iloc[0]
            raise ValueError(f'Failed to load song {song_id}: {self.mert_dir / filename}') from exc
        expected = int((self.label_df.song_id == song_id).sum())
        if len(targets) != expected:
            raise ValueError(f'Features changed since label creation: song {song_id}; recreate labels')
        if features.ndim != 2 or features.shape[1] != 768 or not torch.isfinite(features).all():
            raise ValueError(f'Invalid MERT features: song {song_id}')
        return features, targets, frames


def load_dataset(name, labels=None, feature_dir=None):
    name = dataset_name(name)
    labels = labels or ROOT / f'data/{name}/learning/mert_dynamic_va.parquet'
    meta = json.loads(labels.with_suffix('.json').read_text())
    if meta['dataset'] != name or meta['target_scale'] != [-1, 1]:
        raise ValueError('Dataset/scale mismatch in label metadata')
    df = pd.read_parquet(labels)
    required = ['song_id', 'timestamp', 'valence', 'arousal', 'mert_name']
    if 'fold' in df or df.empty or df[required].isna().any().any():
        raise ValueError('Expected nonempty fold-free MERT labels')
    if df.duplicated(['song_id', 'timestamp']).any() or not np.isfinite(df[['timestamp', 'valence', 'arousal']]).all().all():
        raise ValueError('Duplicate/non-finite labels')
    df['song_id'] = df.song_id.astype(str)
    if (df[['valence', 'arousal']].abs() > 1).any().any():
        raise ValueError('Expected labels on the [-1, 1] scale')
    if len(df) != meta['n_labels'] or df.song_id.nunique() != meta['n_songs']:
        raise ValueError('Labels and metadata counts differ; recreate labels')
    df = pd.concat([g.sort_values('timestamp') for _, g in df.groupby('song_id', sort=False)], ignore_index=True)
    feature_dir = feature_dir or ROOT / f'data/{name}/feature/mert/layer5'
    for filename in df.mert_name.unique():
        path = feature_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing/empty feature file: {path}')
    return FullMERTDataset(df, feature_dir, meta['audio_offset_sec']), meta, labels


def predict_labels(model, features, frames, lengths):
    predictions = model(features, lengths)
    indices = torch.arange(len(features), device=features.device).unsqueeze(1)
    return predictions[indices, frames]


def save_checkpoint(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)
