"""Rank songs by population variance of valid pairwise song CCC values."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def summarize(frame):
    if frame.duplicated(['song_id', 'axis', 'user_i', 'user_j']).any():
        raise ValueError('Duplicate song/axis/pair keys')
    rows = []
    for (song, axis), group in frame.groupby(['song_id', 'axis']):
        ccc = group.ccc.dropna()
        if not np.isfinite(ccc).all():
            raise ValueError('Nonfinite CCC')
        rows.append(dict(song_id=song, axis=axis, n_pairs_total=len(group),
                         n_pairs_valid=len(ccc), ccc_mean=ccc.mean(),
                         ccc_median=ccc.median(), ccc_min=ccc.min(), ccc_max=ccc.max(),
                         ccc_variance=ccc.var(ddof=0) if len(ccc)>=2 else np.nan,
                         ccc_sd=ccc.std(ddof=0) if len(ccc)>=2 else np.nan))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'results/inter_rater_similarity/pairwise_similarity_by_song.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'results/inter_rater_similarity/ccc_variance')
    parser.add_argument('--top', type=int, default=10)
    args = parser.parse_args()
    if args.top < 1:
        parser.error('--top must be positive')
    stats = summarize(pd.read_csv(args.input))
    args.output.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.output/'ccc_variance_by_song.csv', index=False, encoding='utf-8')
    sections = ['# 曲ごとのCCCのペア間分散\n',
                '対象：JaMood 994曲、user_6を除く9名。各曲・各軸で算出済みの36ペアのCCCを使用。',
                'CCCのペア間分散は、有効なCCCの平均からの偏差平方和を有効ペア数で割った値（ddof=0）。NaNは除外し、有効ペアが2未満なら分散はNaN。全994曲をランキング対象とし、同値の場合はsong_id順。',
                '**これは感情ラベル自体の評価者間分散でも、CCCの時間方向の分散でもありません。** 分散が大きい曲ではペアによる一致度の差が大きく、小さい曲ではペアによる差が小さいと解釈します。分散が小さくてもCCC平均が低ければ、全体に一致度が低い可能性があります。36ペアは評価者を共有しており独立ではありません。\n']
    extremes = []
    for axis in ['valence', 'arousal']:
        subset = stats[(stats.axis==axis)&stats.ccc_variance.notna()]
        for label, ascending in [('high',False),('low',True)]:
            table = subset.sort_values(['ccc_variance','song_id'],ascending=[ascending,True]).head(args.top).copy()
            table.insert(0,'rank', range(1,len(table)+1))
            table.insert(1,'selection',label)
            extremes.append(table)
            table.to_csv(args.output/f'{label}_ccc_variance_{axis}.csv',index=False,encoding='utf-8')
            sections.append(f"## {axis.title()}：CCC分散が{'大きい' if label=='high' else '小さい'}{len(table)}曲\n")
            sections.append('| 順位 | song_id | CCC分散 | CCC平均 | CCC最小 | CCC最大 | 有効ペア数 |\n|---:|---:|---:|---:|---:|---:|---:|')
            for row in table.itertuples():
                sections.append(f'| {row.rank} | {row.song_id} | {row.ccc_variance:.6f} | {row.ccc_mean:.6f} | {row.ccc_min:.6f} | {row.ccc_max:.6f} | {row.n_pairs_valid}/{row.n_pairs_total} |')
            sections.append('')
    pd.concat(extremes).to_csv(args.output/'ccc_variance_extremes.csv',index=False,encoding='utf-8')
    sections.append('再実行：`.venv/bin/python scripts/analysis/summarize_ccc_variance.py`\n\n全曲統計：`ccc_variance_by_song.csv`。各分類のCSVと統合CSV `ccc_variance_extremes.csv` も保存。')
    (args.output/'README.md').write_text('\n'.join(sections),encoding='utf-8')
    print(pd.concat(extremes).to_string(index=False))


if __name__ == '__main__':
    main()
