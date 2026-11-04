# JaMood 評価者間分析

## 実行

プロジェクトのルートから実行します。

```bash
.venv/bin/python scripts/analysis/analyze_inter_rater.py
.venv/bin/python -m unittest discover -s scripts/analysis/tests -v
```

依存パッケージは NumPy、pandas、SciPy、Matplotlib です。既存環境にSeabornがないため、描画はMatplotlibで実装しています。

## 対象と入力

- `data/jamood/annotation/processed/cs17_atr_sample_with_annotation_track.csv` の評価者別ラベルを使用。列は `user_id, song_id, timestamp, valence, arousal`。
- `data/jamood/annotation/row/mapping_true.jsonl` の `new_name` と `dataset` でJaMoodを識別。元の1,100曲にはRWC-P 100曲が含まれます。
- ユーザー指定に従い、`data/jamood/annotation/processed/jamood_annotation.csv` のsong_idに対象を限定（994曲）。このCSVの平均済み感情値は使用しません。対応表の1,000曲から除かれるIDは604, 682, 704, 874, 1024, 1062です。
- `analysis/per_rater_annotations/user_mapping.csv` の番号を使用し、ユーザー指定でuser_6を除外。図中の番号は元の1, 2, 3, 4, 5, 7, 8, 9, 10を保持します。
- 元データの−1000〜1000を1000で割り、−1〜1で計算。元スケールへ戻す場合、分散は1,000,000倍、SD/RMSEは1,000倍。PCC/CCCは不変です。

入力は変更しません。出力先は `results/`、既存の同名分析出力は再実行時に更新します。`--output` で別ディレクトリを指定できます。`--song-list` で別のsong_id一覧CSV、`--exclude-users` で除外番号を指定できます。除外なしは値を渡さず `--exclude-users` と指定します。

## 計算規約

- 同じsong_id・同じtimestampを厳密に対応させます。丸め、最近傍結合、補間は行いません。キーの欠損・重複、非数値、無限大、不明なIDはエラーにします。感情値のNaNは許容します。
- 時刻ごとに有効評価者2名以上で母分散（ddof=0）を計算。曲の分散はその時間平均、曲のSDは各時刻のSDの時間平均。有効時刻がない曲はNaNです。各軸の `n_raters_min/mean` は解析に採用した時刻に対して算出。全観測時刻の人数は `timepoint_statistics.csv` に保存します。
- 曲ごと・軸ごと・ペアごとの共通有効時刻でPCC/CCC/RMSEを計算。PCCは2点以上かつ双方非定数、CCCは2点以上かつ分母正のときのみ有効。CCCは母共分散を用いる等価式で計算するため、異なる定数同士はCCC=0、同じ定数同士はNaNです。RMSEは1点以上で計算します。
- ペアの代表値は有効な曲の指標の単純算術平均（曲を等重み）。PCCのFisher変換や曲間のラベル連結は行いません。指標ごとに有効曲数を記録し、NaNを0に置換しません。
- 分布の標準偏差もddof=0。ヒートマップの対角は仕様どおりPCC/CCC=1、RMSE=0（推定値ではありません）。NaNは灰色・NA表記です。RMSEの色範囲は両軸共通です。
- 最大・最小曲が同値の場合はsong_idの小さいものを表示します。

## 出力

要求されたCSVと図を `results/inter_rater_variance/`、`results/inter_rater_similarity/` に保存します。PNGは300dpi、同じ図をベクトルPDFでも保存します。CSVはUTF-8です。

追加出力は、時刻ごとの人数・平均・分散・SD、曲分散の記述統計、Pearson/Spearman係数とp値、極端曲一覧、各ヒートマップの行列CSV、ペア分布の平均・中央値・SDです。`results/data_audit.json` に対象、スケール、欠損、時刻間隔、入力ハッシュ、依存バージョンを保存し、`selected_raters.csv` と `rater_coverage.csv` で評価者対応と曲数を確認できます。

本分析は指定された2種類の記述分析です。軸間差の検定や評価者クラスタリングは実施しません。36ペアは評価者を共有するため独立標本ではなく、ペア数をそのまま独立な反復数として解釈しないでください。
