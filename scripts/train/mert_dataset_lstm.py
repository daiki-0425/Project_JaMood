import pandas as pd
import numpy as np
from pathlib import Path
import torch
import warnings
from torch.utils.data import Dataset, DataLoader

# LogMelDataset クラスの作成
class MERTDataset(Dataset):
    def __init__(
            self,
            label_df,
            mert_dir,
            audio_offset_sec,
    ):
        """
        audio_offset_sec:
            元音声の先頭からカットした長さ (秒)
        """
        self.label_df = label_df
        self.mert_dir = Path(mert_dir)
        self.audio_offset_sec = audio_offset_sec
        self.song_ids = self.label_df["song_id"].unique()

    def __len__(self):
        return len(self.song_ids)

    def __getitem__(self, index):

        """
        features 曲全体の mert : [T, 768]
        targets 各ラベル時刻の valence・arousal : [L, 2]
        frame_nums : [L]
        """

        song_ids = self.song_ids[index]
        song = self.label_df[self.label_df["song_id"] == song_ids].sort_values("timestamp")

        # feature を作成したい
        mert_name = song["mert_name"].iloc[0]
        mert_path = self.mert_dir / mert_name

        with np.load(mert_path) as data:
            feature = data["mert_features"]
            sr = int(data["sr"])
            hop_length = int(data["hop_length"])

            

        # 特徴量を float32 tensor へ [T, 128]
        feature = torch.tensor(feature, dtype=torch.float32)

        # targets を作成して, torch へ
        va = song[["valence", "arousal"]]
        va = torch.tensor(va.to_numpy(dtype=np.float32), dtype=torch.float32) # pandas の dataframe は渡せないので, numpy に変換しておく

        # 正解ラベルに対応するフレームを記憶する
        frame_nums = []
        timestamps = song["timestamp"]

        for timestamp in timestamps:
            frame_num = round(((timestamp / 1000) - self.audio_offset_sec) * sr / hop_length)
            frame_nums.append(frame_num)
        
        frame_nums = torch.tensor(frame_nums, dtype=torch.long)

        # MERT の末尾にはラベル時刻に対応するフレームがない場合がある。
        # バッチの padding を予測値として使わないよう、曲自身の長さで判定する。
        valid_frames = (frame_nums >= 0) & (frame_nums < feature.size(0))
        if not valid_frames.all():
            warnings.warn(
                f"song_id={song_ids}: MERT特徴量の範囲外のラベルを"
                f"{int((~valid_frames).sum())}件除外します "
                f"(frames={feature.size(0)}, "
                f"frame_nums={frame_nums[~valid_frames].tolist()})",
                stacklevel=2,
            )
            frame_nums = frame_nums[valid_frames]
            va = va[valid_frames]
        if frame_nums.numel() == 0:
            raise ValueError(f"song_id={song_ids}: MERT特徴量に対応するラベルがありません")

        return feature, va, frame_nums

        











