import pandas as pd
from pathlib import Path

# pmemo の annotation file を読み込む
input_va_file_path = Path("data/pmemo/annotation/dynamic_annotations_scaled.csv")
df_va = pd.read_csv(input_va_file_path, dtype={"musicId": str})

# pmemo の feature file を読み込む
input_feature_file_path = Path("data/pmemo/feature/emobase988.parquet")
df_feature = pd.read_parquet(input_feature_file_path)

# pmemo の特徴量ファイルを読み込む
input_feature_file_path = Path("data/pmemo/feature/emobase988.parquet")

df_feature = pd.read_parquet(input_feature_file_path)

# output する .parquet ファイル名を定義
output_label_file_path = Path("data/pmemo/feature/label_scaled.parquet")

# frameTime の値を秒からミリ秒へと変換する
df_va["frameTime"] = (df_va["frameTime"] * 1000).astype("int64")

# rename していく
df_va = df_va.rename(
    columns={"musicId": "song_id",
            "frameTime": "timestamp",
            "Arousal(mean)": "arousal",
            "Valence(mean)": "valence"
})

label_df = df_va.merge(
    df_feature,
    on = ["song_id", "timestamp"],
    how="inner",
    validate="one_to_one"
)

label_df = label_df.dropna(subset=["valence", "arousal"])
print(label_df.isna().sum())

label_df.to_parquet(output_label_file_path, index=False)
