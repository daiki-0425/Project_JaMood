import pandas as pd
from pathlib import Path
import librosa
import opensmile

dataset_name = "rwc-p"

input_wav_files_path = Path(f"data/{dataset_name}/audio/processed") 
output_parquet_file_path = Path(f"data/{dataset_name}/feature/emobase988.parquet")

output_parquet_file_path.parent.mkdir(parents=True, exist_ok=True)

# .wavの読み込み
wav_files = sorted(input_wav_files_path.glob("*.wav"))

# opensmileの定義
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.emobase,
    feature_level=opensmile.FeatureLevel.Functionals
)

# 全曲の特徴量
all_song_features = []

for wav_file in wav_files:
    audio, sr = librosa.load(wav_file, sr=None)
    duration = len(audio) / sr

    if duration < 0.5:
        continue

    duration_ms = duration * 1000
    print(wav_file.name, duration)

    song_features = []

    # 曲の上限
    max_timestamp_ms = int((duration_ms + 14500))

    for timestamp_ms in range(15000, max_timestamp_ms + 1, 500):
        start_ms = timestamp_ms - 15000
        end_ms = timestamp_ms - 14500

        start_sample = int(start_ms / 1000 * sr)
        end_sample = int(end_ms / 1000 * sr)

        segment = audio[start_sample:end_sample]

        features = smile.process_signal(segment, sr)
        features["song_id"] = int(wav_file.stem.removeprefix("track_"))
        features["timestamp"] = timestamp_ms

        song_features.append(features)

    # 1曲分をまとめる
    song_features_df = pd.concat(song_features, ignore_index=True)

    # 全曲用のlistに追加
    all_song_features.append(song_features_df)




# 全曲分まとめる
all_features_df = pd.concat(all_song_features, ignore_index=True)

print(all_features_df.shape)
print(all_features_df.isna().sum().sum())
print(all_features_df.head())

# parquet に直す
all_features_df.to_parquet(output_parquet_file_path, index=False)
