"""Train one MERT BiLSTM on all labels, without validation/test splitting."""
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from train_mert_lstm import LSTM, collate_fn
from mert_full_common import DATASETS, ROOT, dataset_name, load_dataset, predict_labels, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--feature-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) <= 0:
        parser.error('epochs and batch-size must be positive')
    args.dataset = dataset_name(args.dataset)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset, metadata, labels = load_dataset(args.dataset, args.labels, args.feature_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    output = args.output_dir or ROOT / f'models/{args.dataset}/mert/layer5/bilstm/full_{datetime.now():%Y%m%d_%H%M%S}'
    output.mkdir(parents=True, exist_ok=False)
    config = dict(dataset=args.dataset, labels=str(labels.resolve()),
                  feature_dir=str(dataset.mert_dir.resolve()), epochs=args.epochs,
                  batch_size=args.batch_size, seed=args.seed, learning_rate=0.001,
                  audio_offset_sec=metadata['audio_offset_sec'], target_scale=[-1, 1],
                  n_songs=len(dataset), n_labels=len(dataset.label_df))
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    model = LSTM().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    print(f'{len(dataset)} songs / {len(dataset.label_df)} labels; device={args.device}; output={output}', flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, count = 0., 0
        for features, targets, frames, mask, lengths in tqdm(loader, desc=f'epoch {epoch}/{args.epochs}'):
            features, targets, frames, mask = [x.to(args.device) for x in (features, targets, frames, mask)]
            optimizer.zero_grad()
            predicted = predict_labels(model, features, frames, lengths)
            loss = torch.nn.functional.mse_loss(predicted[mask], targets[mask])
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            n = int(mask.sum())
            loss_sum += loss.item() * n
            count += n
        history.append(dict(epoch=epoch, train_mse=loss_sum / count, n_labels=count))
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        checkpoint = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                          epoch=epoch, config=config)
        save_checkpoint(checkpoint, output / 'last.pt')
        print(f'epoch {epoch}: train_mse={loss_sum / count:.6f}', flush=True)
    save_checkpoint(checkpoint, output / 'final.pt')
    print(f'Saved final model: {output / "final.pt"}', flush=True)


if __name__ == '__main__':
    main()
