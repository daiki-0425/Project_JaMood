"""Rebuild the 10-rater x 994-song JaMood time-series CSVs from source exports."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [
    ("cs17", "cs17_samples.csv", "cs17_annotations.csv", "cs17_tracks.csv"),
    ("atr", "atr_sample_id.csv", "atr_annotations_id.csv", "atr_track_id_track_name.csv"),
    ("kokawa", "kokawa_samples.csv", "kokawa_annotations.csv", "kokawa_tracks.csv"),
    ("sakiyama", "sakiyama_samples.csv", "sakiyama_annotations.csv", "sakiyama_tracks.csv"),
]
COLUMNS = ["user_id", "song_id", "timestamp", "valence", "arousal", "user_number"]


def main():
    raw = ROOT / "data/jamood/annotation/averaged/raw"
    output_dir = ROOT / "data/jamood/annotation/each rater/processed"
    users = pd.read_csv(ROOT / "analysis/per_rater_annotations/user_mapping.csv")
    if (len(users) != 10 or users.user_id.duplicated().any()
            or set(users.user_number) != set(range(1, 11))):
        raise ValueError("Expected the existing unique user 1-10 mapping")
    with (raw / "mapping_true.jsonl").open(encoding="utf-8") as stream:
        mapping = [json.loads(line) for line in stream if line.strip()]
    names = [entry["new_name"] for entry in mapping]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate new_name in mapping")
    target_names = {
        entry["new_name"] for entry in mapping
        if entry["dataset"] == "JaMood"
        and (ROOT / "data/jamood/audio/raw" / entry["new_name"]).is_file()
    }
    target_ids = {int(Path(name).stem.removeprefix("track_")) for name in target_names}
    if len(target_ids) != 994:
        raise ValueError(f"Expected 994 JaMood songs, got {len(target_ids)}")

    parts = []
    for name, samples_file, annotations_file, tracks_file in SOURCES:
        annotations = pd.read_csv(raw / annotations_file)
        tracks = pd.read_csv(raw / tracks_file)
        samples = pd.read_csv(raw / samples_file)
        tracks["audio_name"] = tracks.path.map(lambda path: Path(path).name)
        tracks = tracks.loc[tracks.audio_name.isin(target_names), ["id", "audio_name"]]
        if tracks.audio_name.duplicated().any():
            raise ValueError(f"Duplicate song paths: {name}")
        annotations = annotations.loc[annotations.user_id.isin(users.user_id)].copy()
        annotations["created_at"] = pd.to_datetime(annotations.created_at, utc=True, errors="raise")
        annotations = annotations.merge(tracks, left_on="track_id", right_on="id",
                                        suffixes=("", "_track"), validate="many_to_one")
        if annotations.duplicated(["user_id", "track_id", "created_at"]).any():
            raise ValueError(f"Ambiguous annotation timestamps: {name}")
        annotations = annotations.sort_values("created_at").drop_duplicates(
            ["user_id", "track_id"], keep="last")
        selected = samples.merge(annotations[["id", "user_id", "audio_name"]],
                                 left_on="annotation_id", right_on="id", validate="many_to_one")
        if set(selected.annotation_id) != set(annotations.id):
            raise ValueError(f"Latest annotations missing samples: {name}")
        selected["song_id"] = selected.audio_name.map(lambda path: int(Path(path).stem.removeprefix("track_")))
        selected["timestamp"] = selected.time_ms / 1000.0
        selected[["valence", "arousal"]] = selected[["valence", "arousal"]] / 1000.0
        selected = selected.merge(users, on="user_id", validate="many_to_one")
        parts.append(selected[COLUMNS])
        print(f"{name}: {selected.user_number.nunique()} raters, {len(selected)} samples", flush=True)

    result = pd.concat(parts, ignore_index=True).sort_values(
        ["user_number", "song_id", "timestamp"]).reset_index(drop=True)
    keys = ["user_number", "song_id", "timestamp"]
    if result.isna().any().any() or result.duplicated(keys).any():
        raise ValueError("Missing values or duplicate rater/song/timestamp")
    if not np.isfinite(result[["timestamp", "valence", "arousal"]]).all().all():
        raise ValueError("Non-finite values")
    if result[["valence", "arousal"]].abs().gt(1).any().any():
        raise ValueError("VA outside [-1, 1]")
    expected_times = np.arange(90) / 2
    for user in range(1, 11):
        frame = result.loc[result.user_number.eq(user)]
        if set(frame.song_id) != target_ids:
            missing = sorted(target_ids - set(frame.song_id))
            raise ValueError(f"user_{user}: incomplete song coverage: {missing}")
        for song_id, song in frame.groupby("song_id"):
            if not np.array_equal(song.timestamp.to_numpy(), expected_times):
                raise ValueError(f"user_{user}/song_{song_id}: expected 90 samples, 0-44.5 seconds")

    # Write only after all ten raters pass the coverage and sample-grid checks.
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "jamood_individual_va.csv"
    result.to_csv(output, index=False)
    for user in range(1, 11):
        result.loc[result.user_number.eq(user)].to_csv(output_dir / f"user_{user}_annotation.csv", index=False)
    print(f"Saved {len(result)} rows: 10 raters x 994 songs x 90 timestamps -> {output}")


if __name__ == "__main__":
    main()
