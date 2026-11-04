import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

csv_dir = Path("analysis/csv/personalized")
feature_dir = Path("data/jamood/feature/mert/layer5")

top50 = pd.read_csv(csv_dir / "arousal_top50.csv")
bottom50 = pd.read_csv(csv_dir / "arousal_bottom50.csv")

output_dir = Path("analysis/figures/personalized/pca")

vectors = []

for song_id in top50["song_id"]:
    feature_path = feature_dir / f"track_{song_id:04d}.npz"

    with np.load(feature_path) as data:
        features = data["mert_features"]

        # 時間方向に平均する
        vector = features.mean(axis=0)
        vectors.append(vector)

    X_top = np.stack(vectors)

vectors = []

for song_id in bottom50["song_id"]:
    feature_path = feature_dir / f"track_{song_id:04d}.npz"

    with np.load(feature_path) as data:
        features = data["mert_features"]

        # 時間方向に平均する
        vector = features.mean(axis=0)
        vectors.append(vector)

    X_bottom = np.stack(vectors)

X = np.concatenate([X_top, X_bottom], axis=0)

pca = PCA(n_components=2)
Z = pca.fit_transform(X)

n_top = len(X_top)



fig = plt.figure(figsize=(8, 6))

plt.scatter(
    Z[:n_top, 0], Z[:n_top, 1],
    label="High disagreement", color="tab:red", alpha=0.7,
)
plt.scatter(
    Z[n_top:, 0], Z[n_top:, 1],
    label="Low disagreement", color="tab:blue", alpha=0.7,
)

ratios = pca.explained_variance_ratio_ * 100
plt.xlabel(f"PC1 ({ratios[0]:.1f}%)")
plt.ylabel(f"PC2 ({ratios[1]:.1f}%)")
plt.title("Arousal disagreement: MERT PCA")
plt.legend()
plt.tight_layout()
fig.savefig(
        output_dir / f"arousal_mert_pca.png",
        dpi=300,
)
plt.close(fig)



