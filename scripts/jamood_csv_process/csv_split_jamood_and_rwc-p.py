import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def split_annotations(df_va, mapping, raw_dirs):
    """new_name を song_id に対応させ、現存する raw 音声の注釈を分割する。"""
    song_ids = {dataset: set() for dataset in raw_dirs}
    mapped_ids = set()
    for entry in mapping:
        dataset = entry["dataset"]
        if dataset not in raw_dirs:
            raise ValueError(f"不明な dataset: {dataset}")
        name = entry["new_name"]
        song_id = int(Path(name).stem.removeprefix("track_"))
        if song_id in mapped_ids:
            raise ValueError(f"new_name の song_id が重複しています: {name}")
        mapped_ids.add(song_id)
        # audio_name は旧音声名。現在の音声名には new_name を使う。
        if (raw_dirs[dataset] / name).is_file():
            song_ids[dataset].add(song_id)

    unknown = set(df_va["song_id"].unique()) - mapped_ids
    if unknown:
        raise ValueError(f"対応表にない song_id: {sorted(unknown)}")
    return {
        dataset: df_va[df_va["song_id"].isin(ids)].copy()
        for dataset, ids in song_ids.items()
    }


def main():
    input_csv = ROOT / "data/jamood/annotation/each rater/raw/user_1_annotations.csv"
    mapping_path = ROOT / "data/jamood/annotation/each rater/raw/mapping_true.jsonl"
    with mapping_path.open(encoding="utf-8") as stream:
        mapping = [json.loads(line) for line in stream if line.strip()]
    df_va = pd.read_csv(input_csv, dtype={"song_id": "int64"})
    dataset_dirs = {"JaMood": "jamood", "RWC-P": "rwc-p"}
    raw_dirs = {
        dataset: ROOT / "data" / directory / "audio/raw"
        for dataset, directory in dataset_dirs.items()
    }
    for directory in raw_dirs.values():
        if not directory.is_dir():
            raise FileNotFoundError(f"入力音声フォルダがありません: {directory}")
    results = split_annotations(df_va, mapping, raw_dirs)
    for dataset, frame in results.items():
        directory = dataset_dirs[dataset]
        output = ROOT / "data" / directory / "annotation/each rater/processed" / f"user_1_annotation.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        print(f"{dataset}: {frame['song_id'].nunique()}曲 / {len(frame)}行 -> {output}")


if __name__ == "__main__":
    main()
