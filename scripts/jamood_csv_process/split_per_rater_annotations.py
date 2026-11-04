import pandas as pd
from pathlib import Path

# input csv file path
input_file_path = Path("data/jamood/annotation/processed/cs17_atr_sample_with_annotation_track.csv")

# annotations df
df_annotations = pd.read_csv(input_file_path)

# output dir
output_dir = Path("analysis/per_rater_annotations")
output_dir.mkdir(parents=True, exist_ok=True)

# user id の取得
user_ids = df_annotations["user_id"].unique()

# ユーザー num と id を mapping する用 user_id : user_num の dict を作成
mappping_num_id = []

# ユーザーごとに csv を抽出する
for i, user_id in enumerate(user_ids, start=1):
    df_per_user_annotation = df_annotations[df_annotations["user_id"] == user_id].copy()
    df_per_user_annotation["user_number"] = i

    df_per_user_annotation["valence"] = df_per_user_annotation["valence"] / 1000
    df_per_user_annotation["arousal"] = df_per_user_annotation["arousal"] / 1000



    mappping_num_id.append({"user_id": user_id, "user_number": i})

    output_path = output_dir / f"user_{i}_annotations.csv"
    df_per_user_annotation.to_csv(output_path, index=False)


# mapping を df にして csv に
pd.DataFrame(mappping_num_id).to_csv(
    output_dir / "user_mapping.csv",
    index=False
)