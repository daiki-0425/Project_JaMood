import pandas as pd
from pathlib import Path
import librosa
import random
import soundfile as sf


# input wav path
input_add_wav_path = Path("music_add/raw")
wav_files = input_add_wav_path.glob("*.wav")

# output wav path
output_add_wav_path = Path("music_add/processed")
output_add_wav_path.mkdir(parents=True, exist_ok=True)


random.seed(42)

# wav を1つずつ読みこむ
for wav_file in wav_files:
    audio, sr = librosa.load(wav_file, sr=None, mono=False)

    n_samples = audio.shape[-1]
    clip_samples = 45 * sr

    start = random.randint(0,n_samples - clip_samples)
    end = start + 45 * sr

    new_audio = audio[:, start:end]

    output_file_path = output_add_wav_path / f"{wav_file.stem}.wav"

    sf.write(output_file_path, new_audio.T, sr, subtype=("PCM_16"))







