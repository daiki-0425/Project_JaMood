import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

input_dir = Path("analysis/per_rater_annotations")

# 全ての csv を把握
df_all = pd.concat(
    [pd.read_csv(path) for path in sorted(
        input_dir.glob("user_*_annotations.csv")
    )],
    ignore_index=True,
)

output_dir = Path("analysis/figures/per_song_va_plot_10_935")
output_dir.mkdir(parents=True, exist_ok=True)

# 曲ごとに, ラベルを付けたユーザーの人数を数える
user_counts = df_all.groupby("song_id")["user_id"].nunique()

target_song_ids = user_counts[user_counts == 10].index

for song_id in target_song_ids:
    df_song = df_all[
        (df_all["song_id"] == song_id) & (df_all["timestamp"].between(14.5,44.5))
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12,7), sharex=True)

    for user_number, df_user in df_song.groupby("user_number"):
        df_user = df_user.sort_values("timestamp")
        color = plt.get_cmap("tab10")(int(user_number) - 1)

        axes[0].plot(
            df_user["timestamp"],
            df_user["valence"],
            color=color,
            label=f"User {user_number}",
            alpha=0.7,
        )

        axes[1].plot(
            df_user["timestamp"],
            df_user["arousal"],
            color=color,
            alpha=0.7,
        )

    axes[0].set_ylabel("Valence")
    axes[1].set_ylabel("Arousal")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_xlim(14.5, 44.5)

    axes[0].legend(ncol=5)
    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Song {song_id}")
    fig.tight_layout()
    fig.savefig(
        output_dir / f"song_{song_id}_va.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


