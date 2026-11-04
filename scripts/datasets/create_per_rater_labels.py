"""評価者1〜10の注釈を分割し、既存のSVR用ラベルと同じ形式で保存する。"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {"JaMood": "jamood", "RWC-P": "rwc-p"}
KEYS = ["song_id", "timestamp"]


def main():
    source = ROOT / "data/jamood/annotation/each rater/raw"
    raw_dirs = {
        name: ROOT / "data" / directory / "audio/raw"
        for name, directory in DATASETS.items()
    }
    for directory in raw_dirs.values():
        if not directory.is_dir():
            raise FileNotFoundError(directory)

    song_ids = {name: set() for name in DATASETS}
    mapped_ids = set()
    with (source / "mapping_true.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            entry = json.loads(line)
            dataset, name = entry["dataset"], entry["new_name"]
            if dataset not in DATASETS:
                raise ValueError(f"不明な dataset: {dataset}")
            song_id = int(Path(name).stem.removeprefix("track_"))
            if song_id in mapped_ids:
                raise ValueError(f"song_id が重複しています: {song_id}")
            mapped_ids.add(song_id)
            if (raw_dirs[dataset] / name).is_file():
                song_ids[dataset].add(song_id)

    annotations = {}
    for user in range(1, 11):
        frame = pd.read_csv(source / f"user_{user}_annotations.csv", dtype={"song_id": "int64"})
        unknown = set(frame["song_id"]) - mapped_ids
        if unknown:
            raise ValueError(f"user_{user}: 対応表にない song_id: {sorted(unknown)}")
        if not frame["user_number"].eq(user).all():
            raise ValueError(f"user_{user}: user_number が一致しません")
        annotations[user] = frame

    for dataset, directory in DATASETS.items():
        base = ROOT / "data" / directory
        features = pd.read_parquet(base / "feature/emobase988.parquet")
        for user, frame in annotations.items():
            split = frame.loc[frame["song_id"].isin(song_ids[dataset])].copy()
            annotation_path = base / "annotation/each rater/processed" / f"user_{user}_annotation.csv"
            labels = split.copy()
            labels["timestamp"] = (labels["timestamp"] * 1000).round().astype("int64")
            labels = labels.merge(features, on=KEYS, how="inner", validate="one_to_one")
            labels = labels.dropna(subset=["valence", "arousal"])
            if labels.empty:
                raise ValueError(f"{dataset}/user_{user}: 結合後のラベルが空です")
            output = base / "feature/each rater" / f"user_{user}" / "label.parquet"
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            split.to_csv(annotation_path, index=False)
            output.parent.mkdir(parents=True, exist_ok=True)
            labels.to_parquet(output, index=False)
            print(
                f"{dataset}/user_{user}: {labels['song_id'].nunique()}曲 / {len(labels)}行 "
                f"(分割前時刻の注釈: {len(split)}行) -> {output.relative_to(ROOT)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
