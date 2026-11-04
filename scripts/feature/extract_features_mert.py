import pandas as pd
from pathlib import Path
import librosa
import numpy as np
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

# dataset name の定義
dataset_name = "jamood"

# 学習済み MERT
model_path = Path("models/MERT-v1-95M")

# file path の定義
input_wav_dir_path = Path(f"data/{dataset_name}/audio/processed")
wav_files = input_wav_dir_path.glob("*.wav")

# 第5層の特徴量の保存先
output_mert_feature_path = Path(f"data/{dataset_name}/feature/mert/layer5")
output_mert_feature_path.mkdir(parents=True, exist_ok=True)

# モデル準備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2FeatureExtractor.from_pretrained(
    str(model_path),
    local_files_only=True,
)

model = AutoModel.from_pretrained(
    str(model_path),
    trust_remote_code=True,
    local_files_only=True
)
model = model.to(device)
model.eval()

target_sr = processor.sampling_rate

# mert 抽出
for wav_file in wav_files:

    # wav から音声を出力
    audio, sr = librosa.load(wav_file, sr=target_sr, mono=True)

    inputs = processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt"
    )

    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    features = outputs.hidden_states[5].squeeze(0)

    output_path = output_mert_feature_path / f"{wav_file.stem}.npz"

    np.savez_compressed(
            output_path,
            mert_features = features.detach().cpu().numpy(),
            sr=sr,
            layer=5,
            hop_length=int(np.prod(model.config.conv_stride)),
    )

    print(f"{wav_file.name}: {tuple(features.shape)} 保存完了")
    del outputs, features, inputs






