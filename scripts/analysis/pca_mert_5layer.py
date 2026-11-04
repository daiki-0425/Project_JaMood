import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

# 特徴量 5つの読み込み
dataset_names = ["jamood", "deam", "memo2496", "pmemo", "rwc-p"]
project_root = Path(__file__).resolve().parents[2]

# 出力ディレクトリの定義
output_dir = project_root / "analysis/figures/PCA"
output_dir.mkdir(parents=True, exist_ok=True)

vectors = []
labels = []

for dataset_name in dataset_names:
    feature_dir = project_root / "data" / dataset_name / "feature/mert/layer5"
    feature_paths = sorted(feature_dir.glob("*.npz"))
    if not feature_paths:
        raise FileNotFoundError(f"特徴量ファイルがありません: {feature_dir}")


    for feature_path in feature_paths:
        with np.load(feature_path) as data:
            features = data["mert_features"]

            # 時間方向に平均する
            vector = features.mean(axis=0)

            vectors.append(vector)
            labels.append(dataset_name)

X = np.stack(vectors)
labels = np.array(labels)

pca = PCA(n_components=2)
Z = pca.fit_transform(X)

print(Z.shape)
print(pca.explained_variance_ratio_)

# 可視化処理

styles = {
    "deam": ("DEAM", "tab:blue"),
    "memo2496": ("MeMo", "tab:orange"),
    "jamood": ("JaMood", "tab:green"),
    "pmemo" : ("PMEmo", "tab:red"),
    "rwc-p": ("RWC-P", "tab:purple"),
}

fig, ax = plt.subplots(figsize=(12,9))

for dataset_name, (display_name, color) in styles.items():
    mask = labels == dataset_name
    points = Z[mask]

    ax.scatter(
        points[:, 0],
        points[:, 1],
        color=color,
        label=display_name,
        s=20,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.3,
    )

ratios = pca.explained_variance_ratio_ * 100
ax.set_xlabel(f"PC1 ({ratios[0]:.2f}%)", fontsize=16)
ax.set_ylabel(f"PC2 ({ratios[1]:.2f}%)", fontsize=16)
ax.legend(loc="upper right", fontsize=13)

fig.tight_layout()

fig.savefig(output_dir / f"mert_layer5_PCA.png",
dpi=150,
bbox_inches="tight",
)

plt.close(fig)
    



