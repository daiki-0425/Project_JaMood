from pathlib import Path
import librosa
import soundfile as sf


dataset_name = "rwc-p"

project_root = Path(__file__).resolve().parents[2]
input_mp3_dir = project_root / f"data/{dataset_name}/audio/raw"
output_mp3_dir = project_root / f"data/{dataset_name}/audio/processed"

output_mp3_dir.mkdir(parents=True, exist_ok=True)

list_mp3_files = sorted(input_mp3_dir.glob("*.wav"))
if not list_mp3_files:
    raise FileNotFoundError(f"入力WAVが見つかりません: {input_mp3_dir}")

print(f"入力WAV: {len(list_mp3_files)}件 / 出力先: {output_mp3_dir}")
processed_count = 0
skipped_count = 0
error_count = 0

for mp3_file in list_mp3_files:
    try:
        audio, sr = librosa.load(mp3_file, sr=None)
        duration = len(audio) / sr
        if duration > 14.5:
            new_audio = audio[int(14.5*sr):]
            output_path = output_mp3_dir / (mp3_file.stem + ".wav")
            sf.write(output_path, new_audio, sr)
            processed_count += 1
        else:
            skipped_count += 1

    except Exception as e:
        error_count += 1
        print(f"ERROR : {mp3_file}")
        print(e)
        continue

print(f"完了: {processed_count}件 / 14.5秒以下: {skipped_count}件 / エラー: {error_count}件")
 


