import matplotlib as plt
from pathlib import Path
import numpy as np
import pandas as pd

# input pmemo csv file path 
input_pmemo_csv_file_path = Path(f"data/pmemo/annotation/dynamic_annotations.csv")

# output pmemo csv file path
output_dir = Path("data/pmemo/annotation")

df_pmemo_va = pd.read_csv(input_pmemo_csv_file_path)

df_pmemo_va["Valence(mean)"] = 2 * df_pmemo_va["Valence(mean)"] - 1
df_pmemo_va["Arousal(mean)"] = 2 * df_pmemo_va["Arousal(mean)"] - 1

# output
df_pmemo_va.to_csv(output_dir / "dynamic_annotations_scaled.csv", index=False)
