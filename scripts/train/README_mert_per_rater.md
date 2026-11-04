# 評価者別 MERT BiLSTM 学習（JaMood）

プロジェクトルートから実行します。

```bash
# 元の評価者別CSVから emobase988 と結合したラベルを作成
.venv/bin/python scripts/datasets/create_per_rater_labels.py
# MERT用の列を抽出し、既存foldと特徴量ファイル名を付与
.venv/bin/python scripts/datasets/create_per_rater_mert_labels.py

# 評価者1、全5 fold、各10 epoch
.venv/bin/python scripts/train/train_mert_lstm.py --user 1
# 評価者1、fold 0のみ
.venv/bin/python scripts/train/train_mert_lstm.py --user 1 --fold 0
# 評価者1〜10を順番に学習
for user in {1..10}; do
    .venv/bin/python scripts/train/train_mert_lstm.py --user "$user" || break
done
```

ラベルは `data/jamood/learning/each rater/user_N/mert_dynamic_vafolds.parquet`、
モデルは `models/jamood/mert/layer5/bilstm/each rater/user_N/best_foldK.pt` に保存します。
同じ評価者・foldを再実行すると、そのモデルは上書きされます。
`--user` を省略すると従来の平均ラベルと保存先を使います。
`--epochs` でepoch数を指定できます（既定10）。

全評価者に `data/folds/jamood/fold_all.csv` の曲単位の分割を使います。
fold Kをテスト、(K+1)%5を検証、残り3つを学習に使います。
評価者6は注釈のある曲だけを含みます。
ラベルのtimestampはミリ秒、VAは元CSVの[-1, 1]を維持します。
MERT layer 5、音声オフセット14.5秒を使い、特徴量の範囲外のラベルは生成時に除外します。
RWC-Pの評価者別ラベルには、この処理でfoldを付与しません。
