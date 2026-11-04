import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# input_dir
input_dir = Path("analysis/per_rater_annotations")

output_dir = Path("analysis/figures/per_user_va_plot")

for i in range(10):
    input_user_annotation_path = input_dir / f"user_{i+1}_annotations.csv"
    df = pd.read_csv(input_user_annotation_path)

    fig, ax = plt.subplots(figsize=(6,6))

    hb = ax.hexbin(
        df["valence"]/1000,
        df["arousal"]/1000,
        gridsize=50,       # 分割の細かさ
        bins="log",       # 点の数を対数スケールで色付け
        mincnt=1,         # 点がない領域は表示しない
        cmap="viridis",
    )

    fig.colorbar(hb, ax=ax, label="Count (log scale)")

    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title(f"User {i+1}: VA distribution")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    fig.tight_layout()

    fig.savefig(output_dir / f"user_{i+1}_va.png",
    dpi=150,
    bbox_inches="tight",
    )

    plt.close(fig)
     

