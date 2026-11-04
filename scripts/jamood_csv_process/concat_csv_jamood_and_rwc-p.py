import pandas as pd
from pathlib import Path

# input csv file パス
input_sample_id_path = Path(f"data/jamood/annotation/averaged/raw/kokawa_samples.csv")
input_track_id_track_name_path = Path(f"data/jamood/annotation/averaged/raw/kokawa_tracks.csv")
input_annotation_id_path = Path(f"data/jamood/annotation/averaged/raw/kokawa_annotations.csv")

df_sample_id = pd.read_csv(input_sample_id_path)
df_track_id_track_name = pd.read_csv(input_track_id_track_name_path)
df_annotation_id = pd.read_csv(input_annotation_id_path)

# output csv file dir
output_csv_dir_path = Path(f"data/jamood/annotation/processed")

# annotation_id から不必要な部分を除去する
df_annotation_id["created_at"] = pd.to_datetime(
    df_annotation_id["created_at"],
    utc=True,
    errors="raise"
)

df_annotation_latest = df_annotation_id.sort_values("created_at").drop_duplicates(
    subset=["user_id", "track_id"],
    keep="last"
)


# sample id と annotation id を結びつける
merge_samples_id_annotation_id = df_sample_id.merge(
    df_annotation_latest,
    left_on="annotation_id",
    right_on="id",
    how="inner", # 左側の sample をすべて残す
    validate="many_to_one", # left は複数ラベル
)


df_track_id_track_name = df_track_id_track_name.rename(columns={"id": "track_id", 
                                                                "duration": "track_duration"})
df_output = merge_samples_id_annotation_id.merge(
    df_track_id_track_name,
    on="track_id",
    how="left",
    validate="many_to_one"
)

df_sorted = df_output.sort_values(
    ["user_id", "track_id", "time_ms"],
    kind="stable",
)



df_sorted["song_id"] = df_sorted["path"].map(
    lambda x : int(Path(x).stem.removeprefix("track_"))
)

df_sorted["timestamp"] = df_sorted["time_ms"] / 1000.0

df = df_sorted[["user_id", "song_id", "timestamp", "valence", "arousal"]]

df = df.sort_values(["user_id", "song_id", "timestamp"]).reset_index(drop=True)

df.to_csv(
    output_csv_dir_path / "kokawa_sample_with_annotation_track.csv",
    index=False
)




