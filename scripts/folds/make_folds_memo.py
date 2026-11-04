import random
from pathlib import Path

import pandas as pd

# fold 数を設定
n_folds = 5

input_song_info_csv_path = Path("data/memo2496/annotation/raw/songs_info_all.csv")
output_csv_path = Path("data/fold/memo2496/fold.csv")

# song_id と file_name のペアを取得する
df = pd.read_csv(input_song_info_csv_path)
selected_df = df[["song_id", "file_name"]]

# 初期順序を固定し、ペアを保ったままシャッフルする
song_records = selected_df.sort_values("song_id").to_dict(orient="records")
random.Random(42).shuffle(song_records)

# 各曲に fold 番号を追加する
for i, row in enumerate(song_records):
    row["fold"] = i % n_folds

# song_id, file_name, fold の3列で保存する
output_df = pd.DataFrame(song_records)
output_csv_path.parent.mkdir(parents=True, exist_ok=True)
output_df.to_csv(output_csv_path, index=False)
