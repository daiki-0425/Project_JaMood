import pandas as pd
from pathlib import Path

# データセット名の定義
dataset_name = "memo2496"

# 各種ファイルの読み込み
input_songs_info_csv = Path(f"data/{dataset_name}/annotation/raw/songs_info_all.csv")
input_arousal_csv = Path(f"data/{dataset_name}/annotation/raw/arousal_all_average.csv")
input_valence_csv = Path(f"data/{dataset_name}/annotation/raw/valence_all_average.csv")
input_feature_parquet = Path(f"data/{dataset_name}/feature/emobase988.parquet")

songs_info_df = pd.read_csv(input_songs_info_csv)
arousal_df = pd.read_csv(input_arousal_csv)
valence_df = pd.read_csv(input_valence_csv)
feature_df = pd.read_parquet(input_feature_parquet)

# output file の定義
output_lable_path = Path(f"data/{dataset_name}/feature/label.parquet") 


# va の timestamp を rename する
valence_df = valence_df.rename(
    columns={
        col: int(col.replace("sample_", "").replace("ms",""))
        for col in valence_df.columns
        if col.startswith("sample_")
    }
)

arousal_df = arousal_df.rename(
    columns={
        col: int(col.replace("sample_", "").replace("ms",""))
        for col in arousal_df.columns
        if col.startswith("sample_")
    }
)

# va を pd.melt により 縦に並べる
melted_arousal = pd.melt(
    arousal_df,
    id_vars = ["song_id"],
    var_name="timestamp",
    value_name="arousal"
)

melted_valence = pd.melt(
    valence_df,
    id_vars = ["song_id"],
    var_name="timestamp",
    value_name="valence"
)

# va を merge する
merged_va_df = melted_valence.merge(
    melted_arousal,
    on=["song_id", "timestamp"],
    how="inner",
    validate="one_to_one"
)

# 曲ID ➣ timestamp の順で sort する
merged_va_df = merged_va_df.sort_values(
    ["song_id", "timestamp"]
).reset_index(drop=True)

# songs_info から file_name を対応付ける
merged_va_df = merged_va_df.merge(
    songs_info_df[["song_id", "file_name"]],
    on=["song_id"],
    how="left",
    validate="many_to_one"
)

# merged_va_df の file name から .mp3 を外す
merged_va_df["file_name"] = merged_va_df["file_name"].str.replace(
    r"\.mp3$", "", regex=True
)

print(merged_va_df["file_name"])

# feature 側でも rename しておく
feature_df = feature_df.rename(columns={"song_id": "file_name"})

label_df = merged_va_df.merge(
    feature_df,
    on=["file_name", "timestamp"], # 同じ曲, 同じ時刻 で feature と対応つける
    how="inner",
    validate="one_to_one"    
)

# 欠損値の確認
label_df = label_df.dropna(subset=["valence", "arousal"])

label_df.to_parquet(output_lable_path, index=False)

