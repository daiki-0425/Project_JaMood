from logmel_dataset_cnn import LogMelDataset
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

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


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # [B, 1, 128, 22] -> [B, 16, 128, 22]

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(
            in_channels =16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(32,2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)

    criterion = nn.MSELoss()    

    # dataset の定義
    dataset_name = "jamood"

    df = pd.read_parquet(f"data/{dataset_name}/learning/logmel_dynamic_vafolds.parquet")

    # fold の処理
    """
    train : valid : test => 60 : 20 : 20
    """

    for i in range(5):

        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = CNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        test_df = df[df["fold"] == i]
        valid_df = df[df["fold"] == (i+1) % 5]
        train_df = df[~df["fold"].isin([i, (i+1) % 5])]

        train_dataset = LogMelDataset(
            label_df=train_df,
            logmel_dir=f"data/{dataset_name}/feature/log_mel",
            window_sec=0.5,
            audio_offset_sec=14.5
        )

        valid_dataset = LogMelDataset(
            label_df=valid_df,
            logmel_dir=f"data/{dataset_name}/feature/log_mel",
            window_sec=0.5,
            audio_offset_sec=14.5
        )

        test_dataset = LogMelDataset(
            label_df=test_df,
            logmel_dir=f"data/{dataset_name}/feature/log_mel",
            window_sec=0.5,
            audio_offset_sec=14.5
        )

        # DataLoader の定義
        loader_options = dict(
            batch_size=32,
            num_workers=4,
            pin_memory=device.type == "cuda",
        )
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
        valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_options)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

        # 保存処理
        best_valid_loss = float("inf")
        save_dir = Path(f"models/{dataset_name}/cnn")
        save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(10):
            print(f"epoch{epoch} start")
            model.train()
            progress = tqdm(train_loader, desc=f"fold {i} epoch {epoch} train")
            for features, targets in progress:
                features = features.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                predictions = model(features)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                progress.set_postfix(loss=f"{loss.item():.4f}")

            model.eval()
            valid_loss_sum = 0.0

            with torch.no_grad():
                for features, targets in tqdm(
                    valid_loader, desc=f"fold {i} epoch {epoch} valid"
                ):
                    features = features.to(device)
                    targets = targets.to(device)

                    predictions = model(features)
                    loss = criterion(predictions, targets)
                    valid_loss_sum += loss.item() * features.size(0) # 平均をかけて

            valid_loss = valid_loss_sum / len(valid_dataset)
            print(f"fold {i} epoch {epoch}: valid_loss={valid_loss:.6f}", flush=True)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(
                    model.state_dict(),
                    save_dir / f"best_fold{i}.pt"
                )

        # test 評価
        # 最良モデルの読み込み
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
            for features, targets in tqdm(test_loader, desc=f"fold {i} test"):
                features = features.to(device)
                predictions = model(features)

                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())


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
        train_valence_mean = train_df["valence"].mean()
        train_arousal_mean = train_df["arousal"].mean()

        baseline_valence_mse = (((y_true[:, 0] - train_valence_mean) ** 2)).mean()
        baseline_valence_rmse = baseline_valence_mse ** 0.5

        baseline_arousal_mse = (((y_true[:, 1] - train_arousal_mean) ** 2)).mean()
        baseline_arousal_rmse = baseline_arousal_mse ** 0.5

        print(f"baseline arousal rmse: {baseline_arousal_rmse}")
        print(f"baseline valence rmse: {baseline_valence_rmse}")
        






                
                



