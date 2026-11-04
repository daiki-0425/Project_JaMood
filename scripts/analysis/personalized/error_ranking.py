import pandas as pd
from pathlib import Path

# 標準偏差が大きい上位50曲　小さい下位50位を pick_up する
df_va = pd.read_csv("data/jamood/annotation/each rater/processed/jamood_individual_va.csv")

# 出力先
output_dir = Path("analysis/csv/personalized")
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

valence.index.name = "song_id"
arousal.index.name = "song_id"

valence.dropna().nlargest(50).to_csv(output_dir / "valence_top50.csv")
valence.dropna().nsmallest(50).to_csv(output_dir / "valence_bottom50.csv")

arousal.dropna().nlargest(50).to_csv(output_dir / "arousal_top50.csv")
arousal.dropna().nsmallest(50).to_csv(output_dir / "arousal_bottom50.csv")

