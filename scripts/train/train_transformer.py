import json
import random
from datetime import datetime
from pathlib import Path

import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

from logmel_dataset_transformer import LogMelDataset
from train_cnn import ccc_score

"""
- B：バッチ内の曲数
- T_max：バッチ内の最大音声フレーム数
- L_max：バッチ内の最大ラベル数
- L_b：選択した曲の有効ラベル数

forward の入力
- features：[B, T_max, 128]
- frame_ranges：[B, L_max, 2]
- audio_padding_mask：[B, T_max]
- label_lengths：[B]

Transformer まで
- 入力射影後の x：[B, T_max, 32]
- 位置情報を加えた x：[B, T_max, 32]
- Transformer 出力の x：[B, T_max, 32]

pooling 中
- 曲番号 i：整数
- 1曲分の frame_range：[L_max, 2]
- 有効区間だけの一覧：[L_b, 2]
- start_frame、end_frame：それぞれスカラー Tensor（shape []）
- 1曲分の特徴量：[T_max, 32]
- 切り出した区間：[end_frame - start_frame, 32]
- 1区間を平均した結果：[32]
- 1曲分を積み重ねた結果：[L_b, 32]
- 全曲分を padding した結果：[B, L_max, 32]

BiLSTM 以降
- BiLSTM 出力を unpack した結果：[B, L_max, 32]
- 出力層の予測：[B, L_max, 2]
- 正解 targets：[B, L_max, 2]
- 損失計算用 label_mask：[B, L_max]

"""

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
    features, targets, frame_ranges = zip(*batch)

    # padding 前の長さ
    audio_lengths = torch.tensor(
        [feature.size(0) for feature in features], dtype=torch.long
    )
    label_lengths = torch.tensor(
        [target.size(0) for target in targets], dtype=torch.long
    )


    features = pad_sequence(
        features, batch_first=True, padding_value=0.0
    )
    targets = pad_sequence(
        targets, batch_first=True, padding_value=0.0
    )
    frame_ranges = pad_sequence(
        frame_ranges, batch_first=True, padding_value=0
    )

    # Ture = 無視する音声の padding
    audio_padding_mask = (
        torch.arange(features.size(1)).unsqueeze(0)>= audio_lengths.unsqueeze(1)
    )

    # True = 有効なラベル
    label_mask = (
        torch.arange(targets.size(1)).unsqueeze(0) < label_lengths.unsqueeze(1)
    )

    return features, targets, frame_ranges, audio_padding_mask, label_mask, label_lengths


# Sinusoidal Positional Encoding の実装
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=32, max_len=100000):
        super().__init__()

        # フレーム位置 : [max_len, 1]
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

        # 次元ごとに異なる周波数: [d_model / 2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        # pe = [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # [1, max_len, d_model]
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError("入力のフレーム数が max_len を超えています")

        return x + self.pe[:, :x.size(1)]


class TransformerBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        # 各フレームの Mel 特徴量を 128 次元から32次元へ変換
        self.input_projection  = nn.Linear(128, 32)
        self.positional_encoding = PositionalEncoding(d_model=32)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.output_layer = nn.Linear(32, 2)


    def forward(self, features, frame_ranges, audio_padding_mask, label_lengths):

        # 入力射影と位置情報
        x = self.input_projection(features) # [B, T_max, 32]
        x = self.positional_encoding(x) # [B, T_max, 32]

        # フレーム間の関係を学習
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=audio_padding_mask) # [B, T_max, 32]

        pooled_songs = []

        # 各ラベルに対応する区間を平均
        for i, frame_range in enumerate(frame_ranges):
            length = label_lengths[i].item()
            pooled_segments = []

            for start_frame, end_frame in frame_range[:length]:
                # pooling して, ラベルの長さと合わせる
                pooled = x[i, start_frame:end_frame, :].mean(dim=0)
                pooled_segments.append(pooled)

            song = torch.stack(pooled_segments, dim=0)
            pooled_songs.append(song)

        # [batch, L_max, 32]
        pooled_batch = pad_sequence(
            pooled_songs,
            batch_first=True,
            padding_value=0.0,
        )

        packed = pack_padded_sequence(
            pooled_batch,
            lengths=label_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, (h_n, c_n) = self.lstm(packed)

        output, output_lengths = pad_packed_sequence(
            packed_output,
            batch_first=True,
        )

        predictions = self.output_layer(output) # [B, L_max, 2]
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

        model = TransformerBiLSTM().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        label_test_df = label_df[label_df["fold"] == i]
        label_valid_df = label_df[label_df["fold"] == (i + 1) % 5]
        label_train_df = label_df[~label_df["fold"].isin([i, (i + 1) % 5])]

        train_dataset = LogMelDataset(
            label_df=label_train_df,
            logmel_dir=logmel_dir,
            audio_offset_sec=14.5,
            window_sec=0.5,
        )

        valid_dataset = LogMelDataset(
            label_df=label_valid_df,
            logmel_dir=logmel_dir,
            audio_offset_sec=14.5,
            window_sec=0.5,
        )

        test_dataset = LogMelDataset(
            label_df=label_test_df,
            logmel_dir=logmel_dir,
            audio_offset_sec=14.5,
            window_sec=0.5,
        )

        # DataLoader の定義
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

        # 保存処理
        best_valid_loss = float("inf")
        save_dir = Path(f"models/{dataset_name}/transformer")
        save_dir.mkdir(parents=True, exist_ok=True)
        loss_history = []

        for epoch in range(10):
            print(f"epoch {epoch} start")
            model.train()
            train_loss_sum = 0.0
            train_label_num = 0

            for features, targets, frame_ranges, audio_padding_mask, label_mask, label_lengths in tqdm(
                train_loader, desc=f"fold {i} epoch {epoch} train"
            ):
                features = features.to(device)
                targets = targets.to(device)
                frame_ranges = frame_ranges.to(device)
                label_mask = label_mask.to(device)
                audio_padding_mask = audio_padding_mask.to(device)

                optimizer.zero_grad()
                predictions = model(features, frame_ranges, audio_padding_mask, label_lengths)

                # padding 部分を除外して損失を計算
                loss = criterion(
                    predictions[label_mask], # [有効ラベル総数, 2]
                    targets[label_mask], # [有効ラベル総数, 2]
                )

                # 逆伝播して重みを更新
                loss.backward()
                optimizer.step()
                label_num = label_mask.sum().item()
                train_loss_sum += loss.item() * label_num
                train_label_num += label_num

            train_loss = train_loss_sum / train_label_num

            # 検証
            model.eval()
            valid_loss_sum = 0.0
            valid_label_num = 0

            with torch.no_grad():
                for features, targets, frame_ranges, audio_padding_mask, label_mask, label_lengths in tqdm(
                    valid_loader, desc=f"fold {i} epoch {epoch} valid"
                ):
                    features = features.to(device)
                    targets = targets.to(device)
                    frame_ranges = frame_ranges.to(device)
                    label_mask = label_mask.to(device)
                    audio_padding_mask = audio_padding_mask.to(device)

                    predictions = model(features, frame_ranges, audio_padding_mask, label_lengths)
                    
                    # padding 部分を除外して損失を計算
                    loss = criterion(
                        predictions[label_mask],
                        targets[label_mask],
                    )
                    valid_loss_sum += loss.item() * label_mask.sum().item()
                    valid_label_num += label_mask.sum().item()

            valid_loss = valid_loss_sum / valid_label_num
            print(
                f"fold {i} epoch {epoch}: "
                f"train_loss={train_loss:.6f}, valid_loss={valid_loss:.6f}",
                flush=True,
            )
            loss_history.append({
                "fold": i,
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            })
            pd.DataFrame(loss_history).to_csv(
                save_dir / f"loss_history_fold{i}.csv", index=False
            )

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
            for features, targets, frame_ranges, audio_padding_mask, label_mask, label_lengths  in tqdm(
                test_loader, desc=f"fold {i} test"
            ):
                features = features.to(device)
                targets = targets.to(device)
                frame_ranges = frame_ranges.to(device)
                label_mask = label_mask.to(device)
                audio_padding_mask = audio_padding_mask.to(device)

                predictions = model(features, frame_ranges, audio_padding_mask, label_lengths)
                

                all_predictions.append(predictions[label_mask].cpu())
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

