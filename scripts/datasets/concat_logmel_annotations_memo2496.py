import pandas as pd
import numpy as np
from pathlib import Path

# dataset名の定義
dataset_name = "memo2496"

# arousal / valence .csv の読み込み
input_arousal_file_path = Path(f"data/{dataset_name}/annotation/raw/arousal_all_average.csv")
input_valence_file_path = Path(f"data/{dataset_name}/annotation/raw/valence_all_average.csv")
input_songs_info_csv = Path(f"data/{dataset_name}/annotation/raw/songs_info_all.csv")

arousal_df = pd.read_csv(input_arousal_file_path, dtype={"song_id": str})
valence_df = pd.read_csv(input_valence_file_path, dtype={"song_id": str})
info_df = pd.read_csv(input_songs_info_csv, dtype={"song_id": str})

# 出力先を設定する
output_file_path = Path(f"data/{dataset_name}/feature/logmel.parquet")
output_file_path.parent.mkdir(parents=True, exist_ok=True)

# time stamp を rename する
arousal_df = arousal_df.rename(
    columns={
        col : int(col.replace("sample_", "").replace("ms", ""))
        for col in arousal_df.columns
        if col.startswith("sample_")
    }
)

valence_df = valence_df.rename(
    columns={
        col : int(col.replace("sample_", "").replace("ms", ""))
        for col in valence_df.columns
        if col.startswith("sample_")
    }
)

# melt する
melted_arousal = pd.melt(
    arousal_df,
    id_vars=["song_id"],
    var_name="timestamp",
    value_name="arousal"
)

melted_valence = pd.melt(
    valence_df,
    id_vars=["song_id"],
    var_name="timestamp",
    value_name="valence"
)

melted_arousal = melted_arousal.sort_values(
    ["song_id", "timestamp"]
).reset_index(drop=True)

melted_valence = melted_valence.sort_values(
    ["song_id", "timestamp"]
).reset_index(drop=True)

merged_va_df = melted_valence.merge(
    melted_arousal,
    on=["song_id", "timestamp"],
    how="inner",
    validate="one_to_one"
)

merged_va_df = merged_va_df.merge(
    info_df[["song_id", "file_name"]],
    on=["song_id"],
    how="left",
    validate="many_to_one",
)

# merged_va_df の file name から .mp3 を外す
merged_va_df["file_name"] = merged_va_df["file_name"].str.replace(
    r"\.mp3$", "", regex=True
)

# 欠損値の除去
merged_va_df = merged_va_df.dropna(subset=["valence", "arousal"])

# log_mel ファイル一覧の取得
input_logmel_file_path = Path(f"data/{dataset_name}/feature/log_mel")
logmel_files = input_logmel_file_path.glob("*.npz")

logmel_records = []

for logmel_file in logmel_files:
    logmel_records.append({"file_name": logmel_file.stem, "logmel_name": logmel_file.name})

logmel_df = pd.DataFrame(logmel_records)

merged_va_df = merged_va_df.merge(
    logmel_df,
    on="file_name",
    how="inner",
    validate="many_to_one"
)

merged_va_df = merged_va_df[
    merged_va_df["timestamp"] >= 15000
]

merged_va_df.to_parquet(output_file_path, index=False)

print(merged_va_df.isna().sum())
print(merged_va_df.head(10))



