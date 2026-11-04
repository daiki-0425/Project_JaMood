import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


class LogMelDataset(Dataset):
    def __init__(self,
                 label_df,
                 logmel_dir,
                 audio_offset_sec,
                 window_sec,
    ):

        self.lable_df = label_df
        self.logmel_dir = Path(logmel_dir)
        self.audio_offset_sec = audio_offset_sec
        self.window_sec = window_sec
        self.song_ids = self.lable_df["song_id"].unique()

    def __len__(self):
        """
        曲数を返す
        """
        return len(self.song_ids)

    def __getitem__(self, index):
        """
        feature [T, 128]
        targets [L, 2]
        pool_ranges [L, 2]
        index に対応した 1つの logmel [T, 128] を取得できたらいい
        """

        song_id = self.song_ids[index]
        song = self.lable_df[self.lable_df["song_id"] == song_id].sort_values("timestamp") 

        # feature を取得したい
        label = song.iloc[0]
        logmel_name = label["logmel_name"]
        logmel_path = self.logmel_dir / logmel_name

        # log-mel の取得
        with np.load(logmel_path) as data:
            logmel = data["log_mel"]
            hop_length = data["hop_length"]
            sr = data["sr"]

        # 特徴量を float32 tensor へ
        feature = torch.tensor(logmel.T, dtype=torch.float32)

        # targets の作成 [L, 2] dtype = torch
        va = song[["valence", "arousal"]]
        va = torch.tensor(va.to_numpy(dtype=np.float32), dtype=torch.float32)

        # 正解ラベルに対応するフレーム区間を記憶する [L, 2] dtype = torch.long
        frame_ranges = []
        timestamps = song["timestamp"]
        window_frame = round(self.window_sec * sr / hop_length)

        for timestamp in timestamps:
            end_frame =  round((timestamp / 1000 - self.audio_offset_sec) * sr / hop_length)
            start_frame = end_frame - window_frame
            if not (0 <= start_frame < end_frame <= feature.shape[0]):
                raise ValueError(
                    f"不正な区間: song_id={song_id}, "
                    f"start={start_frame}, end={end_frame}"
                )
            frame_ranges.append([start_frame, end_frame])

        frame_ranges = torch.tensor(frame_ranges, dtype=torch.long)

        return feature, va, frame_ranges

            


