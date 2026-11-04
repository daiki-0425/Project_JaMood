import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

from logmel_dataset_lstm import LogMelDataset
from train_cnn import ccc_score

# CCC の計算
def ccc_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.mean((y_true - mean_true) ** 2)
    var_pred = np.mean((y_pred - mean_pred) ** 2)

    covariance = np.mean(
        (y_true - mean_true) * (y_pred - mean_pred)
    )

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return 2 * covariance / denominator


def collate_fn(batch):
    features, targets, frame_nums = zip(*batch)

    # padding 前の音声フレーム数（pack 用に CPU に保持）
    lengths = torch.tensor([len(feature) for feature in features], dtype=torch.long)

    # 各曲のラベル数からマスクを作る
    label_masks = []

    for target in targets:
        mask = torch.ones(len(target), dtype=torch.bool)
        label_masks.append(mask)

    label_mask = pad_sequence(
        label_masks,
        batch_first=True,
        padding_value=False,
    )

    features = pad_sequence(
        features,
        batch_first=True,
        padding_value=0.0
    )

    targets = pad_sequence(
        targets,
        batch_first=True,
        padding_value=0.0
    )

    frame_nums = pad_sequence(
        frame_nums,
        batch_first=True,
        padding_value=0
    )

    return features, targets, frame_nums, label_mask, lengths



class LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        """
        入力 : [B, T, 128]
        出力 : [B, T, 2]
        """

        self.lstm = nn.LSTM(
            input_size=128, # 1フレームの mel 特徴量の数
            hidden_size=128, # 隠れ状態の次元数
            num_layers=1, # LSTM の層数
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(256,2)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (h_n, c_n) = self.lstm(packed)
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=x.size(1)
        )
        predictions = self.fc(output)
        return predictions


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset_name = "jamood"

    for i in range(5):

        seed = 42
        random.seed(seed)
        torch.manual_seed(seed)

        # dataset の作成
        label_df = pd.read_parquet(f"data/{dataset_name}/learning/logmel_dynamic_vafolds.parquet")
        logmel_dir = Path(f"data/{dataset_name}/feature/log_mel")

        model = LSTM().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        label_test_df = label_df[label_df["fold"] == i]
        label_valid_df = label_df[label_df["fold"] == (i + 1) % 5]
        label_train_df = label_df[~label_df["fold"].isin([i, (i + 1) % 5])]

        train_dataset = LogMelDataset(
            label_df=label_train_df,
            logmel_dir=logmel_dir,
            audio_offset_sec=14.5
        )

        valid_dataset = LogMelDataset(
            label_df=label_valid_df,
            logmel_dir=logmel_dir,
            audio_offset_sec=14.5
        )

        test_dataset = LogMelDataset(
            label_df=label_test_df,
            logmel_dir=logmel_dir,
            audio_offset_sec=14.5
        )

        # DataLoader の定義
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

        # 保存処理
        best_valid_loss = float("inf")
        save_dir = Path(f"models/{dataset_name}/lstm")
        save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(10):
            print(f"epoch {epoch} start")
            model.train()

            for features, targets, frame_nums, label_mask, lengths in tqdm(
                train_loader, desc=f"fold {i} epoch {epoch} train"
            ):
                features = features.to(device)
                targets = targets.to(device)
                frame_nums = frame_nums.to(device)
                label_mask = label_mask.to(device)

                optimizer.zero_grad()
                predictions = model(features, lengths)

                # 曲番号を作成：[B, 1]
                batch_indices = torch.arange(
                    predictions.size(0), device=device
                ).unsqueeze(1)

                pickup_predictions = predictions[batch_indices, frame_nums]

                # padding 部分を除外して損失を計算
                loss = criterion(
                    pickup_predictions[label_mask],
                    targets[label_mask],
                )

                # 逆伝播して重みを更新
                loss.backward()
                optimizer.step()

            # 検証
            model.eval()
            valid_loss_sum = 0.0
            valid_label_num = 0

            with torch.no_grad():
                for features, targets, frame_nums, label_mask, lengths in tqdm(
                    valid_loader, desc=f"fold {i} epoch {epoch} valid"
                ):
                    features = features.to(device)
                    targets = targets.to(device)
                    frame_nums = frame_nums.to(device)
                    label_mask = label_mask.to(device)

                    predictions = model(features, lengths)
                    
                    # 曲番号を作成：[B, 1]
                    batch_indices = torch.arange(
                        predictions.size(0), device=device
                    ).unsqueeze(1)

                    pickup_predictions = predictions[batch_indices, frame_nums]

                    # padding 部分を除外して損失を計算
                    loss = criterion(
                        pickup_predictions[label_mask],
                        targets[label_mask],
                    )
                    valid_loss_sum += loss.item() * label_mask.sum().item()
                    valid_label_num += label_mask.sum().item()

            valid_loss = valid_loss_sum / valid_label_num
            print(f"fold {i} epoch {epoch}: valid_loss={valid_loss:.6f}", flush=True)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(
                    model.state_dict(),
                    save_dir / f"best_fold{i}.pt"
                )

        # テスト評価
        best_state_dict = torch.load(
            save_dir / f"best_fold{i}.pt",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(best_state_dict)
        model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for features, targets, frame_nums, label_mask, lengths in tqdm(
                test_loader, desc=f"fold {i} test"
            ):
                features = features.to(device)
                targets = targets.to(device)
                frame_nums = frame_nums.to(device)
                label_mask = label_mask.to(device)

                predictions = model(features, lengths)
                
                # 曲番号を作成：[B, 1]
                batch_indices = torch.arange(
                    predictions.size(0), device=device
                ).unsqueeze(1)

                pickup_predictions = predictions[batch_indices, frame_nums]

                all_predictions.append(pickup_predictions[label_mask].cpu())
                all_targets.append(targets[label_mask].cpu())

        y_pred = torch.cat(all_predictions, dim=0).numpy()
        y_true = torch.cat(all_targets, dim=0).numpy()

        mse = (((y_true - y_pred) ** 2)).mean(axis=0)
        rmse = mse ** 0.5
        valence_ccc = ccc_score(y_true[:, 0], y_pred[:, 0])
        arousal_ccc = ccc_score(y_true[:, 1], y_pred[:, 1])
        valence_r2 = r2_score(y_true[:, 0], y_pred[:, 0])
        arousal_r2 = r2_score(y_true[:, 1], y_pred[:, 1])

        print(f"fold{i} の結果")
        print(f"mse : {mse}")
        print(f"rmse : {rmse}")
        print(f"valence_ccc: {valence_ccc}")
        print(f"arousal_ccc: {arousal_ccc}")
        print(f"valence_r2: {valence_r2}")
        print(f"arousal_r2: {arousal_r2}")

        # 平均値予測との比較
        train_valence_mean = label_train_df["valence"].mean()
        train_arousal_mean = label_train_df["arousal"].mean()

        baseline_valence_mse = (((y_true[:, 0] - train_valence_mean) ** 2)).mean()
        baseline_valence_rmse = baseline_valence_mse ** 0.5

        baseline_arousal_mse = (((y_true[:, 1] - train_arousal_mean) ** 2)).mean()
        baseline_arousal_rmse = baseline_arousal_mse ** 0.5

        print(f"baseline arousal rmse: {baseline_arousal_rmse}")
        print(f"baseline valence rmse: {baseline_valence_rmse}")






            




                
                

                






            










        










