import pandas as pd
from pathlib import Path

# input csv file
input_csv_file_path = Path("data/jamood/annotation/processed/cs17_atr_sample_with_annotation_track.csv")

df_label = pd.read_csv(input_csv_file_path)

df_label = df_label.groupby(["song_id", "timestamp"],
                              as_index=False)[["valence", "arousal"]].mean()
df_label[["valence", "arousal"]] /= 1000

# output
output_file_path = Path("data/jamood/annotation/processed/cs17_atr_sample_with_annotation_track_averaged.csv")

df_label.to_csv(output_file_path, index=False)