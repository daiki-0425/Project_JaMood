import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# 評価者別の va.csv の読み込み
df_va = pd.read_csv("data/jamood/annotation/each rater/processed/jamood_individual_va.csv")

# 曲ごとに df_va を取り出していく
# 曲 id リストの取得
song_ids = df_va["song_id"].unique()

# 曲 id 順に 時刻ごとに標準偏差を計算する
for song_id in song_ids:
    song = df_va[df_va["song_id"] == song_id]

    subset = song[song["timestamp"].between(14.5, 44.5)]

    std_per_time = (
        subset.groupby("timestamp")[["valence", "arousal"]].std(ddof=0).sort_index()
    )

    mean_per_time = (
        subset.groupby("timestamp")[["valence", "arousal"]].mean().sort_index()
    )

    print(std_per_time)

    output_dir = Path("analysis/figures/personalized/song_wise_sd_plot")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for ax, label in zip(axes, ["arousal", "valence"]):
        time = mean_per_time.index
        mean = mean_per_time[label]
        std = std_per_time[label]

        ax.plot(time, mean, label="Mean")
        ax.fill_between(
            time,
            mean - std,
            mean + std,
            color="gray",
            alpha=0.3,
            label="Mean ± SD",
        )

        ax.set_ylabel(label.capitalize())
        ax.set_ylim(-1, 1)
        ax.legend()
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(14.5, 44.5)
    fig.suptitle(f"Song {song_id}: Mean ± SD")
    fig.tight_layout()

    output_dir = Path("analysis/figures/personalized/song_wise_sd_plot")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_dir / f"song_{song_id:04d}_mean_sd.png",
        dpi=300,
    )
    plt.close(fig)