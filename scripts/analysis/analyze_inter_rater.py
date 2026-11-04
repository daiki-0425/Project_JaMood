"""Reproducible JaMood inter-rater variance and song-wise similarity analysis."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/jamood-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
AXES = ('valence', 'arousal')
METRICS = ('pcc', 'ccc', 'rmse')


def pair_metrics(x, y):
    """Pairwise complete samples; population covariance/variance for CCC."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    n = len(x)
    result = dict(n_common=n, pcc=np.nan, ccc=np.nan, rmse=np.nan)
    if not n:
        return result
    result['rmse'] = float(np.sqrt(np.mean((x-y)**2)))
    if n >= 2:
        vx, vy = np.var(x), np.var(y)
        cov = np.mean((x-x.mean())*(y-y.mean()))
        if vx > 0 and vy > 0:
            result['pcc'] = float(np.clip(cov / np.sqrt(vx*vy), -1, 1))
        denominator = vx + vy + (x.mean()-y.mean())**2
        if denominator > 0:
            result['ccc'] = float(2*cov / denominator)
    return result


def time_statistics(wide):
    counts = wide.count(axis=1)
    valid = counts >= 2
    variance = wide.var(axis=1, ddof=0).where(valid)
    return pd.DataFrame({'n_raters': counts, 'mean': wide.mean(axis=1).where(valid),
                         'variance': variance, 'sd': np.sqrt(variance)})


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8')


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def distribution(ax, values, title):
    values = pd.Series(values).dropna()
    ax.hist(values, bins=20, color='#477da8', edgecolor='white')
    if len(values):
        ax.axvline(values.mean(), color='#bd452e', label=f'Mean = {values.mean():.4f}')
        ax.axvline(values.median(), color='#272727', linestyle='--', label=f'Median = {values.median():.4f}')
        ax.legend(fontsize=8)
    ax.set(title=title, ylabel='Count')


def load_data(args):
    source = pd.read_csv(args.input)
    required = ['song_id', 'user_id', 'timestamp', *AXES]
    if not set(required) <= set(source.columns):
        raise ValueError(f'Required columns: {required}')
    source = source[required].copy()
    if source[['song_id', 'user_id', 'timestamp']].isna().any().any():
        raise ValueError('Missing song/user/time keys')
    for col in ['song_id', 'timestamp', *AXES]:
        source[col] = pd.to_numeric(source[col], errors='raise')
    if not np.isfinite(source[['song_id', 'timestamp']]).all().all():
        raise ValueError('Nonfinite song/time keys')
    # Match exact seconds, without rounding, interpolation, or nearest joins.
    if source.duplicated(['song_id', 'user_id', 'timestamp']).any():
        raise ValueError('Duplicate song/user/timestamp keys; resolve upstream explicitly')
    if not (source.song_id % 1 == 0).all():
        raise ValueError('Noninteger song IDs')
    records = [json.loads(line) for line in args.mapping.read_text().splitlines() if line.strip()]
    mapping = pd.DataFrame([{'song_id': int(Path(r['new_name']).stem.removeprefix('track_')),
                             'dataset': r['dataset']} for r in records])
    if mapping.song_id.duplicated().any():
        raise ValueError('Duplicate song IDs in mapping')
    if not set(source.song_id) <= set(mapping.song_id):
        raise ValueError('Annotations contain unmapped song IDs')
    ids = sorted(mapping.loc[mapping.dataset == 'JaMood', 'song_id'])
    song_list = pd.read_csv(args.song_list, usecols=['song_id'])
    requested_ids = set(song_list.song_id)
    if not requested_ids <= set(ids):
        raise ValueError('Song list contains IDs outside JaMood mapping')
    excluded_songs = sorted(set(ids) - requested_ids)
    mapped_count = len(ids)
    ids = sorted(requested_ids)
    users = pd.read_csv(args.users)
    if users.user_id.duplicated().any() or users.user_number.duplicated().any():
        raise ValueError('Ambiguous rater mapping')
    if not set(source.user_id) <= set(users.user_id):
        raise ValueError('Unmapped rater IDs')
    users = users[~users.user_number.isin(args.exclude_users)].sort_values('user_number')
    raters = users.user_id.tolist()
    if len(raters) < 2:
        raise ValueError('At least two raters required')
    df = source[source.song_id.isin(ids) & source.user_id.isin(raters)].copy()
    observed_range = df[list(AXES)].agg(['min', 'max']).to_dict()
    if np.isinf(df[list(AXES)]).any().any():
        raise ValueError('Infinite emotion labels')
    if (df[list(AXES)].abs() > 1000).any().any():
        raise ValueError('Labels outside verified native scale [-1000, 1000]')
    missing = df[list(AXES)].isna().sum().to_dict()
    df[list(AXES)] /= 1000.0
    audit = {'input': str(args.input), 'mapping': str(args.mapping),
             'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [args.input, args.mapping, args.users, args.song_list]},
             'source_rows': len(source), 'source_raters': int(source.user_id.nunique()),
             'selected_rows': len(df), 'mapped_jamood_songs': mapped_count, 'selected_songs': len(ids),
             'song_list': str(args.song_list), 'excluded_song_ids': excluded_songs,
             'observed_jamood_songs': int(df.song_id.nunique()), 'selected_raters': len(raters),
             'excluded_user_numbers': args.exclude_users, 'missing_labels': missing,
             'observed_native_ranges': observed_range, 'analysis_scale': 'native / 1000; [-1, 1]',
             'timestamp_policy': 'exact numeric seconds; no rounding or interpolation',
             'variance_ddof': 0, 'pair_aggregation': 'unweighted arithmetic mean of valid song metrics',
             'distribution_sd_ddof': 0,
             'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'matplotlib': matplotlib.__version__, 'scipy': scipy.__version__}}
    return df, ids, users, audit


def analyze(df, ids, users):
    raters = users.user_id.tolist()
    numbers = users.set_index('user_id').user_number.to_dict()
    songs, pairs, times, grids = [], [], [], {}
    grouped = dict(tuple(df.groupby('song_id')))
    for song in ids:
        frame = grouped.get(song, df.iloc[:0])
        song_result = {'song_id': song}
        present = set(frame.user_id)
        for axis in AXES:
            wide = frame.pivot(index='timestamp', columns='user_id', values=axis).reindex(columns=raters).sort_index()
            grids[song, axis] = wide
            ts = time_statistics(wide)
            valid = ts.n_raters >= 2
            song_result.update({f'{axis}_mean_variance': ts.variance.mean(),
                                f'{axis}_mean_sd': ts.sd.mean(),
                                f'{axis}_n_valid_times': int(valid.sum()),
                                f'{axis}_n_times_observed': len(ts),
                                f'{axis}_n_raters_min': ts.loc[valid, 'n_raters'].min(),
                                f'{axis}_n_raters_mean': ts.loc[valid, 'n_raters'].mean()})
            times.append(ts.reset_index().assign(song_id=song, axis=axis))
            for i, j in itertools.combinations(raters, 2):
                if i not in present or j not in present:
                    continue
                pairs.append({'song_id': song, 'axis': axis, 'user_i': numbers[i], 'user_j': numbers[j],
                              **pair_metrics(wide[i].to_numpy(), wide[j].to_numpy())})
        songs.append(song_result)
    by_song = pd.DataFrame(pairs, columns=['song_id','axis','user_i','user_j','n_common',*METRICS])
    summaries = []
    for axis in AXES:
        for i, j in itertools.combinations(users.user_number, 2):
            part = by_song[(by_song.axis == axis) & (by_song.user_i == i) & (by_song.user_j == j)]
            row = {'axis': axis, 'user_i': i, 'user_j': j, 'n_shared_songs': len(part),
                   'n_common_times_total': part.n_common.sum()}
            for metric in METRICS:
                row[f'{metric}_mean'] = part[metric].mean()
                row[f'{metric}_n_valid_songs'] = part[metric].count()
            summaries.append(row)
    return pd.DataFrame(songs), by_song, pd.DataFrame(summaries), pd.concat(times), grids


def plot_results(songs, summary, grids, users, variance_dir, similarity_dir):
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, axis in zip(axs, AXES):
        distribution(ax, songs[f'{axis}_mean_variance'], axis.title())
        ax.set_xlabel('Mean inter-rater variance')
    save_figure(fig, variance_dir / 'variance_distribution')
    valid = songs[['valence_mean_variance', 'arousal_mean_variance']].dropna()
    x, y = valid.to_numpy().T
    correlations = []
    for name, func in [('pearson', pearsonr), ('spearman', spearmanr)]:
        r, p = func(x, y) if len(x) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0 else (np.nan, np.nan)
        correlations.append({'method': name, 'coefficient': r, 'p_value': p, 'n_songs': len(x)})
    save_csv(pd.DataFrame(correlations), variance_dir / 'variance_correlations.csv')
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x, y, s=9, alpha=.4)
    ax.set(xlabel='Valence mean inter-rater variance', ylabel='Arousal mean inter-rater variance',
           title=f"Pearson r = {correlations[0]['coefficient']:.3f}; Spearman r = {correlations[1]['coefficient']:.3f}")
    save_figure(fig, variance_dir / 'variance_scatter')
    extremes = []
    for axis in AXES:
        values = songs.set_index('song_id')[f'{axis}_mean_variance'].dropna()
        if values.empty:
            continue
        for kind, song in [('high', values.idxmax()), ('low', values.idxmin())]:
            extremes.append({'axis': axis, 'selection': kind, 'song_id': song, 'mean_variance': values[song]})
            wide = grids[song, axis]
            ts = time_statistics(wide)
            fig, ax = plt.subplots(figsize=(8, 4))
            for row in users.itertuples():
                ax.plot(wide.index, wide[row.user_id], lw=.8, alpha=.65, label=f'user_{row.user_number}')
            ax.fill_between(wide.index, ts['mean']-ts.sd, ts['mean']+ts.sd, color='black', alpha=.16, label='Mean ± 1 SD')
            ax.plot(wide.index, ts['mean'], color='black', lw=2, label='Mean')
            ax.set(xlabel='Time (s)', ylabel=f'{axis.title()} (native / 1000)',
                   ylim=(-1.5, 1.5), title=f'{kind.title()} variance: song {song} ({values[song]:.4f})')
            ax.legend(ncol=4, fontsize=7)
            save_figure(fig, variance_dir / f'{kind}_variance_{axis}')
    save_csv(pd.DataFrame(extremes), variance_dir / 'extreme_songs.csv')
    numbers = users.user_number.tolist()
    for axis in AXES:
        for metric in METRICS:
            matrix = pd.DataFrame(np.nan, index=numbers, columns=numbers)
            np.fill_diagonal(matrix.values, 0 if metric == 'rmse' else 1)
            for row in summary[summary.axis == axis].to_dict('records'):
                matrix.loc[row['user_i'], row['user_j']] = matrix.loc[row['user_j'], row['user_i']] = row[f'{metric}_mean']
            matrix.to_csv(similarity_dir / f'{metric}_matrix_{axis}.csv', index_label='user_number', encoding='utf-8')
            fig, ax = plt.subplots(figsize=(6, 5))
            cmap = plt.get_cmap('viridis' if metric == 'rmse' else 'RdBu_r').copy()
            cmap.set_bad('#dddddd')
            upper = max(.01, summary.rmse_mean.max()) if metric == 'rmse' else 1
            im = ax.imshow(matrix, cmap=cmap, vmin=0 if metric == 'rmse' else -1, vmax=upper)
            for i in range(len(numbers)):
                for j in range(len(numbers)):
                    v = matrix.iloc[i, j]
                    ax.text(j, i, f'{v:.2f}' if np.isfinite(v) else 'NA', ha='center', va='center', fontsize=7,
                            bbox=dict(facecolor='white', alpha=.65, edgecolor='none', pad=.5))
            ax.set(xticks=range(len(numbers)), yticks=range(len(numbers)), xticklabels=numbers,
                   yticklabels=numbers, xlabel='User number', ylabel='User number', title=f'{axis.title()} — mean song {metric.upper()}')
            fig.colorbar(im, ax=ax)
            save_figure(fig, similarity_dir / f'{metric}_heatmap_{axis}')
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    stats = []
    for i, axis in enumerate(AXES):
        for j, metric in enumerate(METRICS):
            vals = summary.loc[summary.axis == axis, f'{metric}_mean'].dropna()
            distribution(axs[i, j], vals, f'{axis.title()} {metric.upper()} (SD={vals.std(ddof=0):.4f})')
            axs[i, j].set_xlabel(f'Mean song {metric.upper()}')
            stats.append({'axis': axis, 'metric': metric, 'n_pairs': len(vals), 'mean': vals.mean(),
                          'median': vals.median(), 'sd': vals.std(ddof=0)})
    save_figure(fig, similarity_dir / 'similarity_distribution')
    save_csv(pd.DataFrame(stats), similarity_dir / 'similarity_distribution_statistics.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'data/jamood/annotation/processed/cs17_atr_sample_with_annotation_track.csv')
    parser.add_argument('--mapping', type=Path, default=ROOT/'data/jamood/annotation/row/mapping_true.jsonl')
    parser.add_argument('--users', type=Path, default=ROOT/'analysis/per_rater_annotations/user_mapping.csv')
    parser.add_argument('--song-list', type=Path, default=ROOT/'data/jamood/annotation/processed/jamood_annotation.csv', help='CSV providing the requested song_id subset; labels are read from --input')
    parser.add_argument('--exclude-users', nargs='*', type=int, default=[6])
    parser.add_argument('--output', type=Path, default=ROOT/'results')
    args = parser.parse_args()
    df, ids, users, audit = load_data(args)
    vd, sd = args.output/'inter_rater_variance', args.output/'inter_rater_similarity'
    for directory in (vd, sd):
        directory.mkdir(parents=True, exist_ok=True)
    save_csv(users, args.output/'selected_raters.csv')
    coverage = df.groupby('user_id').agg(n_songs=('song_id','nunique'), n_rows=('song_id','size'))
    save_csv(users.merge(coverage, on='user_id', how='left'), args.output/'rater_coverage.csv')
    songs, pairs, summary, times, grids = analyze(df, ids, users)
    save_csv(songs, vd/'song_variance.csv')
    save_csv(times, vd/'timepoint_statistics.csv')
    save_csv(pairs, sd/'pairwise_similarity_by_song.csv')
    save_csv(summary, sd/'pairwise_similarity_summary.csv')
    audit['valid_rater_counts_by_axis'] = {axis: times.loc[times.axis == axis, 'n_raters'].value_counts().sort_index().to_dict() for axis in AXES}
    audit['timestamp_min_max'] = [float(df.timestamp.min()), float(df.timestamp.max())]
    audit['observed_step_counts'] = df.sort_values(['song_id','user_id','timestamp']).groupby(['song_id','user_id']).timestamp.diff().value_counts().sort_index().to_dict()
    (args.output/'data_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    numeric = songs.filter(regex='mean_variance|mean_sd')
    description = numeric.describe().T
    description['std'] = numeric.std(ddof=0)
    save_csv(description.reset_index(names='statistic'), vd/'variance_distribution_statistics.csv')
    plot_results(songs, summary, grids, users, vd, sd)
    print(f'Completed: {len(ids)} songs, {len(users)} raters, {len(pairs)} song/axis/pair rows -> {args.output}')


if __name__ == '__main__':
    main()
