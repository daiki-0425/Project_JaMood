"""JaMood の元曲単位で fold を作成する（標準ライブラリのみ使用）。

mapping_true.jsonl の new_name -> pre_name == track_name -> source_song_id の順で対応付ける。
出力 song_id は new_name の数値部分であり、既存アノテーションと結合できる。
同じ source_song_id の全クリップは必ず同じ fold に割り当てる。
"""

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def required_string(record, key):
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid {key}: {record}")
    return value


def make_folds(tracks, mapping, n_folds=5, seed=42):
    """Return (CSV rows, song_id -> source_song_id); reject incomplete mappings."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    sources = {}
    for track in tracks:
        name = required_string(track, "track_name")
        source = required_string(track, "source_song_id")
        if name in sources:
            raise ValueError(f"Duplicate track_name: {name}")
        sources[name] = source

    song_sources = {}
    seen_names = set()
    seen_pre_names = set()
    for entry in mapping:
        dataset = required_string(entry, "dataset")
        new_name = required_string(entry, "new_name")
        if new_name in seen_names:
            raise ValueError(f"Duplicate new_name: {new_name}")
        seen_names.add(new_name)
        if dataset != "JaMood":
            continue
        pre_name = required_string(entry, "pre_name")
        if pre_name not in sources:
            raise ValueError(f"No track metadata for pre_name: {pre_name}")
        if pre_name in seen_pre_names:
            raise ValueError(f"Duplicate pre_name: {pre_name}")
        seen_pre_names.add(pre_name)
        match = re.fullmatch(r"track_(\d+)\.wav", new_name)
        if match is None:
            raise ValueError(f"Unexpected new_name: {new_name}")
        song_id = int(match.group(1))
        if song_id in song_sources:
            raise ValueError(f"Duplicate song_id: {song_id}")
        song_sources[song_id] = sources[pre_name]

    missing = sources.keys() - seen_pre_names
    if missing:
        raise ValueError(f"Tracks missing from mapping: {sorted(missing)}")
    unique_sources = sorted(set(song_sources.values()))
    if len(unique_sources) < n_folds:
        raise ValueError("Not enough source songs for the requested folds")

    # クリップではなく、重複を除いた元曲をシャッフルして分割する。
    random.Random(seed).shuffle(unique_sources)
    source_folds = {source: i % n_folds for i, source in enumerate(unique_sources)}
    rows = [
        {"song_id": song_id, "fold": source_folds[song_sources[song_id]]}
        for song_id in sorted(song_sources)
    ]

    # CSV 書き出し前にも元曲の fold 重複を検査する。
    assigned = defaultdict(set)
    for row in rows:
        assigned[song_sources[row["song_id"]]].add(row["fold"])
    if any(len(folds) != 1 for folds in assigned.values()):
        raise ValueError("Source song leakage across folds")
    return rows, song_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    raw = ROOT / "data/jamood/annotation/row"
    parser.add_argument("--tracks", type=Path, default=raw / "track_0001_1000.jsonl")
    parser.add_argument("--mapping", type=Path, default=raw / "mapping_true.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/folds/jamood/fold_all.csv")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows, song_sources = make_folds(
        read_jsonl(args.tracks), read_jsonl(args.mapping), args.n_folds, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["song_id", "fold"])
        writer.writeheader()
        writer.writerows(rows)
    clip_counts = Counter(row["fold"] for row in rows)
    source_sets = defaultdict(set)
    for row in rows:
        source_sets[row["fold"]].add(song_sources[row["song_id"]])
    print(f"Saved {len(rows)} clips / {len(set(song_sources.values()))} source songs: {args.output}")
    for fold in range(args.n_folds):
        print(f"fold {fold}: {len(source_sets[fold])} source songs, {clip_counts[fold]} clips")


if __name__ == "__main__":
    main()
