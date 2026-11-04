import pandas as pd
import shutil
import random
from pathlib import Path
import json

"""
JaMood/Audio から 1100曲読み取って JaMood と RWC-P に分割する
"""

dataset_name = "jamood"

# 必要ファイルの読み込み
input_jamood_wav_dir_path = Path(f"data/{dataset_name}/audio/row")
input_json_file_path = Path(f"data/{dataset_name}/annotation/row/mapping.jsonl")

# 保存場所の定義
output_jamood_wav_dir_path = Path(f"data/jamood/audio/processed")
output_rwc_p_wav_dir_path = Path(f"data/rwc-p/audio/processed")


wav_files = input_jamood_wav_dir_path.glob("*.wav")
with open(input_json_file_path, "r", encoding="utf-8") as f:
    mapping = [json.loads(line) for line in f if line.strip()] # 空行をスキップ

# str型で jamood と rwc-p に対応するファイル名を取得する
jamood = []
rwc_p = []

for item in mapping:
    if item["dataset"] == "JaMood":
        jamood.append(item["new_name"])
    else:
        rwc_p.append(item["new_name"])

# copy していくぅ
for wav_file in wav_files:
    if wav_file.name in jamood:
        shutil.copy2(wav_file, output_jamood_wav_dir_path / wav_file.name)
    else:
        shutil.copy2(wav_file, output_rwc_p_wav_dir_path / wav_file.name)

print("complete")





