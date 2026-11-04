"""Create JaMood per-rater MERT labels using the existing song-level folds."""

from pathlib import Path

import numpy as np
import pandas as pd

from create_mert_labels_full import feature_info

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["song_id", "timestamp"]


def main():
    base = ROOT / "data/jamood"
    folds = pd.read_csv(ROOT / "data/folds/jamood/fold_all.csv", dtype={"song_id": "int64"})
    if folds.song_id.duplicated().any() or not folds.fold.isin(range(5)).all():
        raise ValueError("Invalid song-level folds")

    # Metadata is shared by all raters; inspect each feature file only once.
    metadata = {}
    for user in range(1, 11):
        source = base / f"feature/each rater/user_{user}/label.parquet"
        df = pd.read_parquet(source, columns=KEYS + ["valence", "arousal", "user_number"])
        if df.empty or df.isna().any().any() or df.duplicated(KEYS).any():
            raise ValueError(f"Empty, missing or duplicate labels: {source}")
        if not df.user_number.eq(user).all():
            raise ValueError(f"Wrong rater: {source}")
        if not np.isfinite(df[KEYS + ["valence", "arousal"]]).all().all():
            raise ValueError(f"Non-finite labels: {source}")
        if df[["valence", "arousal"]].abs().gt(1).any().any():
            raise ValueError(f"Labels outside [-1, 1]: {source}")
        df = df.merge(folds[["song_id", "fold"]], on="song_id", how="left", validate="many_to_one")
        if df.fold.isna().any():
            raise ValueError(f"Missing folds: {source}")
        df["mert_name"] = df.song_id.map(lambda song: f"track_{song:04d}.npz")
        parts = []
        for name, song in df.groupby("mert_name", sort=False):
            if name not in metadata:
                metadata[name] = feature_info(base / "feature/mert/layer5" / name)
            n, sr, hop = metadata[name]
            frames = np.rint((song.timestamp.to_numpy() / 1000 - 14.5) * sr / hop)
            valid = (frames >= 0) & (frames < n)
            if not valid.any():
                raise ValueError(f"No labels within MERT frames: {name}")
            parts.append(song.loc[valid])
        result = pd.concat(parts, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
        if set(result.fold) != set(range(5)):
            raise ValueError(f"Not all five folds represented: {source}")
        output = base / f"learning/each rater/user_{user}/mert_dynamic_vafolds.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output, index=False)
        print(f"user_{user}: {result.song_id.nunique()} songs, {len(result)} labels, "
              f"{len(df) - len(result)} out-of-range labels excluded -> {output.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
