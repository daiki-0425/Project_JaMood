import random
import pandas as pd
from pathlib import Path


# fold 数の設定
n_folds = 5

dataset_name = "memo2496"
input_wav_file_path = Path(f"data/{dataset_name}/audio/processed")
output_csv_file_path = Path(f"data/fold/{dataset_name}/fold.csv")

# 曲名を保存しておくリスト
song_names = []

listed_wav_files = list(input_wav_file_path.glob("*.wav"))

for wav_file in listed_wav_files:
    song_names.append(wav_file.stem)

# ソートにより初期順序固定
song_names = sorted(song_names)

# shuffle の実行
random.Random(42).shuffle(song_names)

# fold 用の song_id, fold の dict を格納する list を定義
folds = []

for i, song_name in enumerate(song_names):
    row = {"song_id": song_name, "fold": i % n_folds}
    folds.append(row)

df = pd.DataFrame(folds)

output_csv_file_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_csv_file_path, index=False)