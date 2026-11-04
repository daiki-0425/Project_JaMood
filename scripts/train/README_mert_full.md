# MERT 全データ学習と外部データ評価

プロジェクトルートで仮想環境を有効にして実行する。
既存の fold 学習コード・ラベル・モデルとは別に保存する。
`memo` は `memo2496` の別名として指定できる。

## 1. fold なしラベル

```bash
python scripts/datasets/create_mert_labels_full.py
```

JaMood、PMEmo、DEAM、MeMo、RWC-P の
`data/<dataset>/learning/mert_dynamic_va.parquet` と同名の `.json` を作成する。
特定のデータだけ作る場合は `--datasets jamood rwc-p` を指定する。

既存の MERT 用ラベルから全 fold の行を取得する。MERT 用ラベルがまだない
RWC-P は既存の `feature/logmel.parquet` のラベルとファイル名対応を利用する。
log-mel 特徴量の抽出や学習は実行しない。
対象列は `song_id, timestamp, valence, arousal, mert_name` で、fold 列は含めない。

MERT の第5層・768次元の特徴量を使用する。時刻はミリ秒。
元音声の切り出しオフセットは JaMood / DEAM / RWC-P が14.5秒、PMEmo が15秒、
MeMo が14秒。PMEmo / MeMo は複数の元音声と加工済み音声の波形照合で確認した。
各ラベルの JSON にオフセットを記録し、学習・評価コードはそれを読む。
各特徴量のフレーム数・サンプリング周波数・hop length を読み、対応フレームが
範囲外のラベルを除外する。除外数と曲ごとの内訳は JSON に記録する。
特徴量の欠損、空ファイル、不正なメタデータ、全ラベルが範囲外の曲はエラーにする。
ラベル作成時の検査は特徴量のヘッダーとメタデータが対象で、行列の有限値検査は学習・評価の読込時に行う。

全データセットの教師値を **[-1, 1]** にそろえる。
PMEmo は既存ラベルと生のアノテーションの一致を確認し、`2*x-1` を適用する。
既存の fold 用ラベルには変更を加えない。
「全データ」は既存ラベル表に含まれる全曲のうち、MERT 特徴量に対応する全時刻を意味する。
元音声・元アノテーションの全時刻を追加するものではない。

## 2. JaMood 全データで10エポック学習

```bash
python scripts/train/train_mert_lstm_full.py \
  --dataset jamood \
  --epochs 10 \
  --output-dir models/jamood/mert/layer5/bilstm/full_10epochs
```

`--dataset pmemo`、`deam`、`memo2496` でも同様に学習できる。
エポック数の既定値は10。検証・テスト分割をせず、新規初期化した BiLSTM を学習する。
既存モデルと同じ隠れ次元128、双方向1層、Adam、学習率0.001、バッチサイズ2を用いる。
MERT 自体は更新せず、抽出済み特徴量を入力する。
RWC-P をテストに使う場合、学習データには含めない。

保存先には `config.json`、`history.csv`、各エポック終了時の `last.pt`、
指定エポック完了時の `final.pt` が作成される。検証によるモデル選択は行わない。
チェックポイントにはモデル・optimizer・エポック・設定を保存するが、自動再開は実装していない。
上書きを防ぐため、明示した出力ディレクトリは未作成の場所を指定する。
`--output-dir` を省略すると日時付きの新しい保存先を使う。
抽出完了後に学習し、学習・評価中は特徴量を書き換えない。

## 3. RWC-P 全100曲でテスト

```bash
python scripts/train/test_mert_lstm_full.py \
  --dataset rwc-p \
  --checkpoint models/jamood/mert/layer5/bilstm/full_10epochs/final.pt \
  --output-dir outputs/mert_full/jamood_to_rwc-p
```

全有効時刻をまとめた Valence / Arousal の RMSE・R²・CCC を
`metrics.csv`、`metrics.json` に保存する。
曲ID・時刻・正解値・予測値は `predictions.csv` に保存する。
評価でパラメータ更新や最良エポックの選択は行わない。
この全データ学習コードが作成したチェックポイントを指定する。

両コードとも `--device cpu` / `--device cuda`、`--batch-size`、
`--labels`、`--feature-dir` を指定できる。
カスタムラベルには対応する `.json` も必要。

## 4. JaMood → PMEmo / DEAM / MEMO2496 の一括評価

```bash
python scripts/train/evaluate_jamood_cross_dataset.py \
  --output-dir outputs/mert_cross_dataset/jamood_to_external
```

既定では `models/jamood/mert/layer5/bilstm/full_10epochs/final.pt` を使い、
3データセットそれぞれの全有効ラベルを評価する。再学習・fold 分割・
評価先でのモデル選択は行わない。全データ学習コードで保存した JaMood の
完了済みチェックポイントのみ対応する。別のモデルは `--checkpoint` で指定する。

`--datasets pmemo deam` で対象を絞れる。`--device cpu` / `--device cuda`、
`--batch-size 2`、`--data-root data` も指定できる。
入力は `<data-root>/<dataset>/learning/mert_dynamic_va.parquet` と同名 JSON、
`<data-root>/<dataset>/feature/mert/layer5/` の抽出済み特徴量。
手順1で作成したスケール統一済みラベルとデータセット固有のオフセットを使用する。

出力先には新しいディレクトリを指定する。省略時は日時付きで自動作成する。
各データセットのサブディレクトリに `metrics.csv`、`metrics.json`、
`predictions.csv`、出力先直下に比較用の `summary.csv` と実行情報 `run.json` を保存する。
比較表はデータセット×Valence/Arousal の6行で、RMSE・R²・CCC と評価曲数・
ラベル数・ラベル作成時の除外数を記録する。指標は各データセットの全有効時刻を
まとめて計算し、曲単位やデータセット間の平均は取らない。
途中で失敗した場合は終了コードを非ゼロにし、`run.json` に失敗状態と
完了済みデータセットを残す。比較表には完了分のみが入る。

## 5. 任意の学習元からのクロスデータセット評価

```bash
# DEAM → JaMood / MEMO2496 / PMEmo
python scripts/train/evaluate_cross_dataset.py --train-datasets deam

# MEMO2496 → JaMood / DEAM / PMEmo（memo は memo2496 の別名）
python scripts/train/evaluate_cross_dataset.py --train-datasets memo

# PMEmo → JaMood / DEAM / MEMO2496
python scripts/train/evaluate_cross_dataset.py --train-datasets pmemo

# 4つの学習元すべて × 自分以外の3データセット（計12方向）
python scripts/train/evaluate_cross_dataset.py --train-datasets all \
  --output-dir outputs/mert_cross_dataset/all_models
```

`--train-datasets deam memo pmemo` のように複数の学習元も指定できる。
`--datasets jamood pmemo` で評価先を絞れる。学習元と評価先が同じ組み合わせは除外する。
既定のモデルは JaMood / DEAM / PMEmo が `full_10epochs/final.pt`、
MEMO2496 が `full_20261025_052316/final.pt`（いずれも
`models/<dataset>/mert/layer5/bilstm/` 配下）。
別の学習済みモデルを使う場合は学習元を1つ指定し、`--checkpoint` を渡す。
チェックポイント内の学習元・スケール・完了エポックを検査する。

結果は `<output-dir>/<train_dataset>/<test_dataset>/` 配下に保存し、
直下の `summary.csv` に全組み合わせをまとめる。12方向なら Valence / Arousal の計24行。
`run.json` にモデルのパス、評価組み合わせ、完了済みの組み合わせ、実行状態を記録する。
従来の `evaluate_jamood_cross_dataset.py` も引き続き使用でき、
その場合の結果ディレクトリは従来どおり `<output-dir>/<test_dataset>/`。

時間方向の評価方法は既存処理と同じで、各ラベル時刻に最も近い MERT フレームの
予測を比較する。MEMO2496 は1秒間隔のラベルを使うが、1秒窓での平均化は行わない。

## 6. JaMood の fold モデルによる曲別 CCC

```bash
python scripts/train/evaluate_jamood_per_song_folds.py \
  --device cuda --output-dir outputs/jamood_per_song_ccc_folds_20261029
```

`models/jamood/mert/layer5/bilstm/best_fold{0..4}.pt` を使う。
JaMood はモデルと同じ番号のテスト fold の曲だけを評価するため、
全5 fold で各曲の評価は1回。検証 fold は `(model_fold + 1) % 5`、
残り3 fold が学習用という元の学習コードの分割に対応する。
RWC-P は各 fold モデルで外部100曲すべてを評価する。
RWC-P 自体に fold を割り当てるわけではない。
`--model-dir`、`--fold-labels`、`--folds 0 1`、`--batch-size` で変更できる。
全データ学習の `final.pt` ではなく fold ごとの state dict に対応する。

`per_song_ccc.csv` に評価先・モデル fold・曲ID・ラベル数・Valence/Arousal の
CCC と RMSE を保存する。`fold_summary.csv` は曲別 CCC の単純平均・中央値・
標本標準偏差と、全時刻を結合した CCC (`pooled_ccc`) を区別して保存する。
各曲の時系列の標準偏差も保存する。ラベル2件未満または CCC の分母が0の曲は
CCC を欠損値にし、平均・中央値から除外して件数を記録する。

`<評価先>/fold<番号>/` に予測時系列と曲別結果を保存し、RWC-P はさらに
`rwc-p_valence_ccc_by_fold.csv` / `rwc-p_arousal_ccc_by_fold.csv` に
曲×モデル fold の表を保存する。出力先は新しいディレクトリを指定する。
各曲の全有効時刻を使用し、平滑化はしない。
