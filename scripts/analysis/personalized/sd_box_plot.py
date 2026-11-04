import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


df_va = pd.read_csv("data/jamood/annotation/each rater/processed/jamood_individual_va.csv")

# 出力先
output_dir = Path("analysis/figures/personalized/box_plot")
output_dir.mkdir(parents=True, exist_ok=True)

song_ids = df_va["song_id"].unique()

valence_std_mean_per_songs = []
arousal_std_mean_per_songs = []

for song_id in song_ids:
    song = df_va[(df_va["song_id"] == song_id) & (df_va["timestamp"].between(14.5, 44.5))]

    per_time_std = song.groupby("timestamp")[["valence", "arousal"]].std(ddof=0).sort_index()

    song_va_mean = per_time_std.mean()

    valence_std_mean_per_song = song_va_mean["valence"]
    arousal_std_mean_per_song = song_va_mean["arousal"]

    valence_std_mean_per_songs.append(valence_std_mean_per_song)
    arousal_std_mean_per_songs.append(arousal_std_mean_per_song)


# 曲ID付きのSeriesに変換
valence = pd.Series(valence_std_mean_per_songs, index=song_ids, name="valence_std_mean")
arousal = pd.Series(arousal_std_mean_per_songs, index=song_ids, name="arousal_std_mean")

fig, ax = plt.subplots(figsize=(6,5))

ax.boxplot([valence.dropna(), arousal.dropna()])
ax.set_xticks([1,2], ["Valence", "Arousal"])
ax.set_ylabel("Mean standard deviation across raters")

fig.tight_layout()
fig.savefig(
    output_dir / "valence_arousal_mean_std_boxplot.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)