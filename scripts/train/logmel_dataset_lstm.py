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
            audio_offset_sec,
    ):
        """
        audio_offset_sec:
            元音声の先頭からカットした長さ (秒)
        """
        self.label_df = label_df
        self.logmel_dir = Path(logmel_dir)
        self.audio_offset_sec = audio_offset_sec
        self.song_ids = self.label_df["song_id"].unique()

    def __len__(self):
        return len(self.song_ids)

    def __getitem__(self, index):

        """
        features 曲全体の log-mel : [T, 128]
        targets 各ラベル時刻の valence・arousal : [L, 2]
        frame_nums : [L]
        """

        song_ids = self.song_ids[index]
        song = self.label_df[self.label_df["song_id"] == song_ids].sort_values("timestamp")

        # feature を作成したい
        logmel_name = song["logmel_name"].iloc[0]
        logmel_path = self.logmel_dir / logmel_name

        with np.load(logmel_path) as data:
            logmel = data["log_mel"]
            sr = int(data["sr"])
            hop_length = int(data["hop_length"])

            

        # 特徴量を float32 tensor へ [T, 128]
        feature = torch.tensor(logmel.T, dtype=torch.float32)

        # targets を作成して, torch へ
        va = song[["valence", "arousal"]]
        va = torch.tensor(va.to_numpy(dtype=np.float32), dtype=torch.float32) # pandas の dataframe は渡せないので, numpy に変換しておく

        # 正解ラベルに対応するフレームを記憶する
        frame_nums = []
        timestamps = song["timestamp"]

        for timestamp in timestamps:
            frame_position = ((timestamp / 1000) - self.audio_offset_sec) * sr / hop_length
            # 末尾の丸め上げのみ補正する。負の時刻や特徴量の範囲を
            # 超える時刻は、データの不整合として GPU に渡す前に停止する。
            if not np.isfinite(frame_position) or not 0 <= frame_position < len(feature):
                raise ValueError(
                    f"song_id={song_ids}, timestamp={timestamp}, "
                    f"frame_position={frame_position}, frames={len(feature)}, "
                    f"audio_offset_sec={self.audio_offset_sec}"
                )
            frame_num = min(round(frame_position), len(feature) - 1)
            frame_nums.append(frame_num)
        
        frame_nums = torch.tensor(frame_nums, dtype=torch.long)

        return feature, va, frame_nums

        











