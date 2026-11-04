import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

# LogMelDataset クラスの作成
class LogMelDataset(Dataset):
    def __init__(
            self,
            label_df,
            logmel_dir,
            window_sec,
            audio_offset_sec,
    ):
        """
        audio_offset_sec:
            元音声の先頭からカットした長さ (秒)
        """
        self.label_df = label_df
        self.logmel_dir = Path(logmel_dir)
        self.window_sec = window_sec
        self.audio_offset_sec = audio_offset_sec

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, index):
        label = self.label_df.iloc[index] # label_df の中で何行目を取得するか
        logmel_name = label["logmel_name"]
        timestamp = label["timestamp"]
        logmel_path = self.logmel_dir / logmel_name
    
    
        # log-melの取得
        with np.load(logmel_path) as data:
            logmel = data["log_mel"]
            hop_length = data["hop_length"]
            sr = data["sr"]
    
            
        # timestamp を frame に変換する
        window_frame = round(self.window_sec * sr / hop_length)
        end_frame = round((timestamp / 1000 - self.audio_offset_sec) *  sr / hop_length )
        start_frame = end_frame - window_frame
    
        logmel_extract = logmel[:, start_frame:end_frame]

        # 特徴量を float32 の Tensor にする
        feature = torch.tensor(logmel_extract, dtype=torch.float32)
        feature = feature.unsqueeze(0)

        # valence, arousal も tensor化する
        target = torch.tensor(
            [label["valence"], label["arousal"]],
            dtype = torch.float32,
        )

        return feature, target
        



