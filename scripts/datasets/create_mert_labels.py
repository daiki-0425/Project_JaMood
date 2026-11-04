from pathlib import Path

import pandas as pd


dataset_name = "jamood"

input_path = Path(f"data/{dataset_name}/learning/logmel_dynamic_vafolds.parquet")
output_path = Path(f"data/{dataset_name}/learning/mert_dynamic_vafolds.parquet")
mert_dir = Path(f"data/{dataset_name}/feature/mert/layer5")


def main():
    df = pd.read_parquet(input_path)
    if "fold" not in df.columns:
        raise ValueError(f"fold 列がありません: {input_path}")
    if "logmel_name" not in df.columns:
        raise ValueError(f"logmel_name 列がありません: {input_path}")
    if "mert_name" in df.columns:
        raise ValueError(f"mert_name 列がすでに存在します: {input_path}")
    if df["logmel_name"].isna().any():
        raise ValueError("logmel_name 列に欠損値があります")

    # ファイル名は共通なので、列名だけを変更する。
    df = df.rename(columns={"logmel_name": "mert_name"})

    missing = [
        name for name in df["mert_name"].unique()
        if not (mert_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"MERT特徴量が {len(missing)} 曲分ありません: {missing[:10]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"保存完了: {output_path} ({len(df)} 行)")


if __name__ == "__main__":
    main()
