import pandas as pd
from pathlib import Path

# pmemo の annotation file を読み込む
input_va_file_path = Path("data/rwc-p/annotation/processed/rwc-p_annotation.csv")
df_va = pd.read_csv(input_va_file_path, dtype={"song_id": "int64"})

df_va["timestamp"] = (
    df_va["timestamp"] * 1000
).round().astype("int64")

# pmemo の feature file を読み込む
input_feature_file_path = Path("data/rwc-p/feature/emobase988.parquet")
df_feature = pd.read_parquet(input_feature_file_path)

# output する .parquet ファイル名を定義
output_label_file_path = Path("data/rwc-p/feature/label.parquet")



label_df = df_va.merge(
    df_feature,
    on = ["song_id", "timestamp"],
    how="inner",
    validate="one_to_one"
)

label_df = label_df.dropna(subset=["valence", "arousal"])
print(label_df.isna().sum())

label_df.to_parquet(output_label_file_path, index=False)
