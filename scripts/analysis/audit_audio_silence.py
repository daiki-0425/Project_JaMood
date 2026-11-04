"""Audit exact digital silence and quiet 0.5 s windows in selected JaMood audio."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]


def measure(path):
    x,sr=sf.read(path,always_2d=True,dtype='float64')
    if not len(x):
        raise ValueError(f'Empty audio: {path}')
    active=np.any(x!=0,axis=1)
    indices=np.flatnonzero(active)
    rms=np.sqrt(np.mean(x*x))
    peak=np.max(np.abs(x))
    step=round(sr*.5)
    windows=[x[i:i+step] for i in range(0,len(x),step)]
    quiet=np.array([np.sqrt(np.mean(w*w))<=.001 for w in windows])
    lengths=np.array([len(w) for w in windows])
    return dict(sample_rate=sr,channels=x.shape[1],duration_s=len(x)/sr,
                rms_dbfs=20*np.log10(rms) if rms else -np.inf,
                peak_dbfs=20*np.log10(peak) if peak else -np.inf,
                all_zero=not active.any(),exact_zero_fraction=1-active.mean(),
                leading_zero_s=indices[0]/sr if len(indices) else len(x)/sr,
                trailing_zero_s=(len(x)-1-indices[-1])/sr if len(indices) else len(x)/sr,
                quiet_window_fraction=float(lengths[quiet].sum()/len(x)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/audio_silence')
    args=parser.parse_args()
    ids=sorted(pd.read_csv(ROOT/'data/jamood/annotation/processed/jamood_annotation.csv').song_id.unique())
    rows=[]
    for song in ids:
        for variant in ['raw','processed']:
            path=ROOT/f'data/jamood/audio/{variant}/track_{song:04d}.wav'
            row=dict(song_id=song,track=f'track_{song:04d}',variant=variant,path=str(path),exists=path.is_file())
            if path.is_file():
                row.update(measure(path))
            rows.append(row)
    args.output.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(args.output/'audio_silence_by_track.csv',index=False,encoding='utf-8')
    summary=[]
    for variant in ['raw','processed']:
        d=frame[(frame.variant==variant)&frame.exists]
        conditions={'all_zero':d.all_zero.eq(True),
                    'trailing_exact_silence_ge_10s':d.trailing_zero_s>=10,
                    'trailing_exact_silence_ge_29s':d.trailing_zero_s>=29,
                    'quiet_windows_ge_95pct':d.quiet_window_fraction>=.95,
                    'whole_file_rms_le_minus60dbfs':d.rms_dbfs<=-60}
        for label,mask in conditions.items():
            subset=d[mask].sort_values(['trailing_zero_s','song_id'],ascending=[False,True])
            subset.to_csv(args.output/f'{variant}_{label}.csv',index=False,encoding='utf-8')
            summary.append(dict(variant=variant,criterion=label,n_tracks=len(subset),
                                tracks=' '.join(subset.track)))
    out=pd.DataFrame(summary)
    out.to_csv(args.output/'silence_summary.csv',index=False,encoding='utf-8')
    print(out.to_string(index=False))
    print('Missing files:',frame.loc[~frame.exists,['song_id','variant']].to_dict('records'))
    print('Reference:',frame[frame.song_id==1057].to_string(index=False))


if __name__=='__main__':
    main()
