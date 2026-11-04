"""Create fold-free MERT labels on a common [-1, 1] VA scale."""

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ['jamood', 'pmemo', 'deam', 'memo2496', 'rwc-p']
OFFSETS = {'jamood': 14.5, 'pmemo': 15.0, 'deam': 14.5, 'memo2496': 14.0, 'rwc-p': 14.5}


def feature_info(path):
    # Read the NPY header without decompressing the entire feature matrix.
    with zipfile.ZipFile(path) as archive:
        with archive.open('mert_features.npy') as stream:
            version = np.lib.format.read_magic(stream)
            reader = { (1, 0): np.lib.format.read_array_header_1_0,
                       (2, 0): np.lib.format.read_array_header_2_0 }[version]
            shape, _, dtype = reader(stream)
    with np.load(path, allow_pickle=False) as data:
        sr, hop = int(data['sr']), int(data['hop_length'])
        if int(data['layer']) != 5:
            raise ValueError(f'Expected MERT layer 5: {path}')
    if len(shape) != 2 or shape[0] == 0 or shape[1] != 768 or dtype.kind != 'f' or min(sr, hop) <= 0:
        raise ValueError(f'Invalid MERT shape/metadata: {path}: {shape}')
    return shape[0], sr, hop


def create_labels(name, root=ROOT, offset=None):
    offset = OFFSETS[name] if offset is None else offset
    base = root / 'data' / name
    source = base / 'learning/mert_dynamic_vafolds.parquet'
    if not source.exists():
        source = base / 'feature/logmel.parquet'
    df = pd.read_parquet(source).rename(columns={'logmel_name': 'mert_name'})
    df = df[['song_id', 'timestamp', 'valence', 'arousal', 'mert_name']].copy()
    df['song_id'] = df['song_id'].astype(str)
    if df.empty or df.isna().any().any() or df.duplicated(['song_id', 'timestamp']).any():
        raise ValueError(f'Empty, missing or duplicate labels: {source}')
    for col in ['timestamp', 'valence', 'arousal']:
        df[col] = pd.to_numeric(df[col], errors='raise')
    if not np.isfinite(df[['timestamp', 'valence', 'arousal']]).all().all():
        raise ValueError(f'Non-finite labels: {source}')
    if name == 'pmemo':
        # Verify against raw annotations to avoid accidentally scaling twice.
        raw = pd.read_csv(base / 'annotation/dynamic_annotations.csv', dtype={'musicId': str})
        raw = raw.rename(columns={'musicId': 'song_id', 'Valence(mean)': 'valence', 'Arousal(mean)': 'arousal'})
        raw['timestamp'] = (raw['frameTime'] * 1000).round().astype('int64')
        check = df.merge(raw[['song_id', 'timestamp', 'valence', 'arousal']],
                         on=['song_id', 'timestamp'], how='left', suffixes=('', '_raw'), validate='one_to_one')
        for col in ['valence', 'arousal']:
            if not np.allclose(check[col], check[col + '_raw']):
                raise ValueError('PMEmo source differs from raw [0, 1] annotations')
        df[['valence', 'arousal']] = 2 * df[['valence', 'arousal']] - 1
    if (df[['valence', 'arousal']].abs() > 1).any().any():
        raise ValueError(f'Labels outside [-1, 1]: {source}')
    parts, excluded = [], []
    for index, (song_id, song) in enumerate(df.groupby('song_id', sort=False)):
        if song.mert_name.nunique() != 1:
            raise ValueError(f'Multiple features for song {song_id}')
        path = base / 'feature/mert/layer5' / song.mert_name.iloc[0]
        try:
            n, sr, hop = feature_info(path)
        except Exception as exc:
            raise ValueError(f'Cannot read feature metadata: {path}') from exc
        frames = np.rint((song.timestamp.to_numpy() / 1000 - offset) * sr / hop).astype('int64')
        valid = (frames >= 0) & (frames < n)
        if not valid.any():
            raise ValueError(f'No valid labels: {path}')
        parts.append(song.loc[valid].sort_values('timestamp'))
        if not valid.all():
            excluded.append({'song_id': song_id, 'count': int((~valid).sum())})
        if (index + 1) % 500 == 0:
            print(f'{name}: checked {index + 1} songs', flush=True)
    result = pd.concat(parts, ignore_index=True)
    output = base / 'learning/mert_dynamic_va.parquet'
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix('.tmp.parquet')
    result.to_parquet(temp, index=False)
    temp.replace(output)
    metadata = dict(dataset=name, source=str(source), audio_offset_sec=offset,
                    target_scale=[-1, 1], pmemo_rescaled=name == 'pmemo',
                    n_source_labels=len(df), n_labels=len(result), n_songs=result.song_id.nunique(),
                    excluded_labels=len(df) - len(result), exclusions=excluded)
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'{name}: {metadata["n_songs"]} songs, {len(result)} labels, '
          f'{metadata["excluded_labels"]} out-of-range labels excluded -> {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS + ['memo'], default=DATASETS)
    args = parser.parse_args()
    for name in args.datasets:
        create_labels('memo2496' if name == 'memo' else name)
