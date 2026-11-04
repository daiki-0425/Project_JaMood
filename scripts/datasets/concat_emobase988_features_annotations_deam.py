import pandas as pd
from pathlib import Path

# deam の annotation file を読み込む
input_arousal_file_path = Path("data/deam/annotation/dynamics/dynamic (averaged per song)/arousal.csv")
input_valence_file_path = Path("data/deam/annotation/dynamics/dynamic (averaged per song)/valence.csv")

df_arousal = pd.read_csv(input_arousal_file_path, dtype={"song_id": str})
df_valence = pd.read_csv(input_valence_file_path, dtype={"song_id": str})

# deam の特徴量ファイルを読み込む
input_feature_file_path = Path("data/deam/feature/emobase988.parquet")

df_feature = pd.read_parquet(input_feature_file_path)

# output する .parquet ファイル名を定義
output_label_file_path = Path("data/deam/feature/label.parquet")

# timestamp から sample と ms を除去する
df_arousal = df_arousal.rename(
    columns={
        col: int(col.replace("sample_", "").replace("ms", ""))
        for col in df_arousal.columns
        if col.startswith("sample_")
    }
)

df_valence = df_valence.rename(
    columns={
        col: int(col.replace("sample_", "").replace("ms", ""))
        for col in df_valence.columns
        if col.startswith("sample_")
    }
)

# pd.melt で song_id 以外の時刻列を縦に並べる
melted_arousal_df = pd.melt(
    df_arousal,
    id_vars=["song_id"],
    var_name="timestamp",
    value_name="arousal"
)

melted_valence_df = pd.melt(
    df_valence,
    id_vars=["song_id"],
    var_name="timestamp",
    value_name="valence"
)

# 曲ID ➣ 時刻の順で sort

melted_arousal_df = melted_arousal_df.sort_values(
    ["song_id", "timestamp"]
).reset_index(drop=True)

melted_valence_df = melted_valence_df.sort_values(
    ["song_id", "timestamp"]
).reset_index(drop=True)

# valence と arousal を結合する
merged_arousal_valence_df = melted_arousal_df.merge(
    melted_valence_df,
    on=["song_id", "timestamp"],
    how="inner",
    validate="one_to_one"
)

merged_arousal_valence_df = merged_arousal_valence_df.dropna(subset=["valence", "arousal"])

label_df = merged_arousal_valence_df.merge(
    df_feature,
    on = ["song_id", "timestamp"],
    how="inner",
    validate="one_to_one"
)

label_df = label_df.dropna(subset=["valence", "arousal"])

print(label_df.isna().sum())

label_df.to_parquet(output_label_file_path, index=False)


