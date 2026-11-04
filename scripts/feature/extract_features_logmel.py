import pandas as pd
from pathlib import Path
import librosa
import numpy as np

# dataset名の定義
dataset_name = "jamood"

# wav の読み込み
input_wav_files_path = Path(f"data/{dataset_name}/audio/processed")
wav_files = sorted(input_wav_files_path.glob("*.wav"))

# log-mel の出力先の定義
output_log_mels_path = Path(f"data/{dataset_name}/feature/log_mel")
output_log_mels_path.mkdir(parents=True, exist_ok=True)

for wav_file in wav_files:

    # 音声波形とサンプリング周波数の取得
    audio, sr = librosa.load(wav_file, sr=22050)

    duration = len(audio) / sr
    if duration < 0.5:
        print(f"skip: {wav_file.name} (0.5秒未満)")
        continue

    # mel の計算

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        power=2.0
    )

    # 対数化する
    log_mel = librosa.power_to_db(mel, ref=1.0, top_db=80)

    output_path = output_log_mels_path / f"{wav_file.stem}.npz"

    np.savez_compressed(
        output_path,
        log_mel=log_mel,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        center=True,
        ref=1.0,
        top_db=80
    )

    print(f"{wav_file.stem} is down")


