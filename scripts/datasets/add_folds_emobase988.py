import pandas as pd
from pathlib import Path

dataset_name = "jamood"
input_va_parquet_path = Path(f"data/{dataset_name}/feature/label.parquet")
input_fold_csv_path = Path(f"data/folds/{dataset_name}/fold_all.csv")

output_with_fold_path = Path(f"data/{dataset_name}/learning/emobase988_dynamic_vafolds.parquet")

va_df = pd.read_parquet(input_va_parquet_path)
fold_df = pd.read_csv(input_fold_csv_path, dtype={"song_id": "int64"})

merged_df = va_df.merge(
    fold_df[["song_id", "fold"]],
    on="song_id",
    how="left",
    validate="many_to_one"
)

print(merged_df.isna().sum())

merged_df.to_parquet(output_with_fold_path, index=False)
