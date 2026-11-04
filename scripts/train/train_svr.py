import argparse

import pandas as pd
import numpy as np
from pathlib import Path


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib

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

parser = argparse.ArgumentParser(description="5-fold SVR training")
parser.add_argument(
    "--max-iter", type=int, default=-1,
    help="Maximum optimization iterations per model; -1 means unlimited (default).",
)
args = parser.parse_args()
if args.max_iter != -1 and args.max_iter <= 0:
    parser.error("--max-iter must be a positive integer or -1")
print(f"SVR max_iter={args.max_iter} (per target and fold)", flush=True)

# 対象のデータセット名の定義
dataset_name = "jamood"

# 保存処理の定義
save_dir = Path(f"data/{dataset_name}/svr")
save_dir.mkdir(parents=True, exist_ok=True)

# fold付き Parquet の読み込み
input_with_fold_parquet_path = Path(f"data/{dataset_name}/learning/emobase988_dynamic_vafolds.parquet")
df = pd.read_parquet(input_with_fold_parquet_path)

# fold を考慮した 5-fold 検証
for i in range(0,5):
    print(f"\n===== fold {i} =====", flush=True)

    train = df[df["fold"] != i]
    test = df[df["fold"] == i]

    # train の読み込み
    X_train = train.drop(columns=["song_id","timestamp","fold","arousal", "valence"])
    y_train_arousal = train["arousal"]
    y_train_valence = train["valence"]

    # test の読み込み
    X_test = test.drop(columns=["song_id","timestamp","fold","arousal", "valence"])
    y_test_arousal = test["arousal"]
    y_test_valence = test["valence"]

    # arousal の Pipeline を作成する
    svr_arousal = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(verbose=True, max_iter=args.max_iter))
    ])

    print("arousal 学習開始")

    # train の基準で valid を予測する
    svr_arousal.fit(X_train, y_train_arousal)

    # 保存処理
    joblib.dump(svr_arousal, save_dir / f"arousal_fold{i}.joblib")

    y_test_pred_arousal = svr_arousal.predict(X_test)

    print("arousal 学習終了")

    # 予測誤差の確認
    mae_arousal = mean_absolute_error(y_test_arousal, y_test_pred_arousal)
    mse_arousal = mean_squared_error(y_test_arousal, y_test_pred_arousal)
    rmse_arousal = mse_arousal ** 0.5
    ccc_arousal = ccc_score(y_test_arousal, y_test_pred_arousal)
    r2_arousal = r2_score(y_test_arousal, y_test_pred_arousal) 

    print(f"mae_arousal: {mae_arousal}")
    print(f"mse_arousal: {mse_arousal}")
    print(f"rmse_arousal: {rmse_arousal}")
    print(f"ccc: {ccc_arousal}")
    print(f"r2: {r2_arousal}")

    # valence の Pipeline を作成する
    svr_valence = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(verbose=True, max_iter=args.max_iter))
    ])

    print("valence 学習開始")

    # train の基準で valid を予測する
    svr_valence.fit(X_train, y_train_valence)

    # 保存処理
    joblib.dump(svr_valence, save_dir / f"valence_fold{i}.joblib")

    y_test_pred_valence = svr_valence.predict(X_test)

    # 予測誤差の確認
    mae_valence = mean_absolute_error(y_test_valence, y_test_pred_valence)
    mse_valence = mean_squared_error(y_test_valence, y_test_pred_valence)
    rmse_valence = mse_valence ** 0.5
    ccc_valence = ccc_score(y_test_valence, y_test_pred_valence)
    r2_valence = r2_score(y_test_valence, y_test_pred_valence) 
    

    print("valence 学習終了")
    print(f"mae_valence: {mae_valence}")
    print(f"mse_valence: {mse_valence}")
    print(f"rmse_valence: {rmse_valence}")
    print(f"ccc: {ccc_valence}")
    print(f"r2: {r2_valence}")


