import pandas as pd
import numpy as np
from pathlib import Path

# dataset名の定義
dataset_name = "pmemo"

# arousal / valence .csv の読み込み
# SVRと共通の曲・時刻・ラベルを使う
va_df = pd.read_parquet(
    f"data/{dataset_name}/feature/label.parquet",
    columns=["song_id", "timestamp", "valence", "arousal"],
)
va_df["song_id"] = va_df["song_id"].astype(str)


# 出力先を設定する
output_file_path = Path(f"data/{dataset_name}/feature/logmel.parquet")
output_file_path.parent.mkdir(parents=True, exist_ok=True)

# 欠損値の除去
va_df = va_df.dropna(subset=["valence", "arousal"])

# log_mel ファイル一覧の取得
input_logmel_file_path = Path(f"data/{dataset_name}/feature/log_mel")
logmel_files = input_logmel_file_path.glob("*.npz")

logmel_records = []

for logmel_file in logmel_files:
    logmel_records.append({"song_id": logmel_file.stem, "logmel_name": logmel_file.name})

logmel_df = pd.DataFrame(logmel_records)

va_df = va_df.merge(
    logmel_df,
    on="song_id",
    how="inner",
    validate="many_to_one"
)

va_df.to_parquet(output_file_path, index=False)

print(va_df.isna().sum())
print(va_df.head(10))



