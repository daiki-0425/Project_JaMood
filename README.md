# Project JaMood

音楽の Valence / Arousal を対象とする、特徴量抽出・モデル学習・評価・分析用の研究コードです。
JaMood、DEAM、PMEmo、MeMo2496、RWC-P を扱うスクリプトを含みます。

## このリポジトリに含むもの

- `scripts/feature/`: log-mel、emobase、MERT の特徴量抽出
- `scripts/datasets/`、`scripts/jamood_csv_process/`: データ整形・ラベル作成
- `scripts/folds/`: 学習・評価の分割作成
- `scripts/train/`: 学習と評価
- `scripts/analysis/`: 分析処理のソースコード
- `requirements*.txt`: Python の依存関係

音源、CSV、アノテーション、抽出済み特徴量、モデル重み、DB、解析結果・図、
圧縮ファイルは同梱しません。`music-va-app-static/` と `music-va-app-dynamics/` も対象外です。
これらは `.gitignore` で除外し、手元のファイルは保持します。

## 実行の準備

1. 必要なデータと事前学習モデルは、それぞれの提供元の条件を確認し、別途用意してください。
2. Python の仮想環境を作成して `requirements.txt` の依存をインストールしてください。
   PyTorch は利用環境に応じて導入してください。`requirements-gpu.txt` は CUDA 用の設定です。
3. データは `data/<dataset>/`、MERT は `models/MERT-v1-95M/` に配置します。
   各スクリプトの入力ディレクトリと引数を確認してください。

一部スクリプトはデータセット名やパスをソース内で設定します。
データ・モデルを同梱しないため、clone のみでは学習や評価は実行できません。
依存関係全体の新規環境での動作確認は未実施です。

詳細は [MERT の学習と評価](scripts/train/README_mert_full.md)、
[評価者別の学習](scripts/train/README_mert_per_rater.md)、
[評価者間の分析](scripts/analysis/README_inter_rater.md) を参照してください。

## 著作権と配布方針

音源・データセット・モデル・依存ライブラリには、それぞれの提供元の利用条件が適用されます。
このリポジトリは、それらの利用・再配布の許諾を提供するものではありません。
加工音源、特徴量、アノテーション、学習済みモデル、図表を追加で配布する場合も、
元データの条件と評価者情報の扱いを個別に確認してください。

本プロジェクトのソースコードの配布ライセンスは未設定です。
第三者のコード・素材の権利確認と、権利者によるライセンス選択を経て設定してください。
参考: [GitHub のライセンス設定ガイド](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)。

## GitHub 登録時の注意

`.gitignore` は、既に追跡されているファイルや過去のコミットからデータを削除するものではありません。
既存の履歴をアップロードする場合は、履歴も別途確認してください。
初回登録では、除外設定を適用したソースコードだけの新規リポジトリを使用する方針です。
ライセンスと公開範囲の確認が終わるまでは非公開で管理します。
