import pandas as pd
from pathlib import Path
import numpy as np
from xgboost import  XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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


# dataset を定義
dataset_name = "jamood"

# 学習データの読み込み
input_train_data_file_path = Path(f"data/{dataset_name}/learning/emobase988_dynamic_vafolds.parquet")
df = pd.read_parquet(input_train_data_file_path)

save_dir = Path(f"data/{dataset_name}/xgboost")
save_dir.mkdir(parents=True, exist_ok=True)

df.columns = (
    df.columns
    .str.replace("[", "_", regex=False)
    .str.replace("]", "", regex=False)
)

# 学習, 検証, 評価データに分割する
for i in range(5):

    print(f"====== fold{i} ======")
    train = df[df["fold"] != i]
    test = df[df["fold"] == i]

    # 特徴量とラベルの準備
    X_train = train.drop(columns=["fold", "valence", "arousal", "timestamp", "song_id"])
    y_train_arousal = train["arousal"]
    y_train_valence = train["valence"]

    X_test = test.drop(columns=["fold", "valence", "arousal", "timestamp", "song_id"])
    y_test_arousal = test["arousal"]
    y_test_valence = test["valence"]
 

    # XGBRegessor のインスタンスの作成
    xgb_arousal = XGBRegressor(
        objective="reg:squarederror",
        max_depth=4,
        learning_rate=0.05,
        n_estimators=500,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="hist",
        eval_metric="rmse",
        random_state=43
    )

    xgb_valence = XGBRegressor(
        objective="reg:squarederror",
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="hist",
        eval_metric="rmse",
        random_state=42
    ) 

    # 学習 phase
    xgb_arousal.fit(
        X_train,
        y_train_arousal,
        verbose=50
    )

    # 予測 arousal
    y_test_pred_arousal = xgb_arousal.predict(X_test)

    # 評価指標 CCC, RMSE, R2
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

    xgb_valence.fit(
        X_train,
        y_train_valence,
        verbose=50
    )

    # 予測 arousal
    y_test_pred_valence = xgb_valence.predict(X_test)

    # 評価指標 CCC, RMSE, R2
    mae_valence = mean_absolute_error(y_test_valence, y_test_pred_valence)
    mse_valence = mean_squared_error(y_test_valence, y_test_pred_valence)
    rmse_valence = mse_valence ** 0.5
    ccc_valence = ccc_score(y_test_valence, y_test_pred_valence)
    r2_valence = r2_score(y_test_valence, y_test_pred_valence) 


    # 保存の処理
    xgb_arousal.save_model(save_dir / f"arousal_fold{i}.ubj")
    xgb_valence.save_model(save_dir / f"valence_fold{i}.ubj")

    print("valence 学習終了")
    print(f"mae_valence: {mae_valence}")
    print(f"mse_valence: {mse_valence}")
    print(f"rmse_valence: {rmse_valence}")
    print(f"ccc: {ccc_valence}")
    print(f"r2: {r2_valence}")

