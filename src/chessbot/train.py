"""Train the policy net by behavioral cloning on the player's moves.

Loss is cross-entropy from the policy logits to the move the player actually played.
Validation metric is **top-1 move-match accuracy** (how often the bot's top move equals
the player's move) plus top-3 - this is the direct measure of imitation quality. Train/val
are split by *game* so positions from one game never straddle the split.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .encoding import NUM_MOVES
from .model import build_model


def _load_samples(user: str):
    path = config.samples_path(user)
    if not path.exists():
        raise FileNotFoundError(f"No samples at {path}; run chessbot.dataset first.")
    data = np.load(path)
    return data["X"], data["y"], data["game_ids"]


def _split_by_game(game_ids: np.ndarray, val_fraction: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    unique = np.unique(game_ids)
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * val_fraction))
    val_games = set(unique[:n_val].tolist())
    is_val = np.array([g in val_games for g in game_ids])
    return ~is_val, is_val


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    top3 = logits.topk(3, dim=1).indices
    top1_correct = (top3[:, 0] == target).sum().item()
    top3_correct = (top3 == target.unsqueeze(1)).any(dim=1).sum().item()
    n = target.size(0)
    return top1_correct / n, top3_correct / n


def train(user: str, cfg: config.Config = config.DEFAULT, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X, y, game_ids = _load_samples(user)
    train_mask, val_mask = _split_by_game(game_ids, cfg.data.val_fraction)

    def make_loader(mask: np.ndarray, shuffle: bool) -> DataLoader:
        xt = torch.from_numpy(X[mask]).float()
        yt = torch.from_numpy(y[mask]).long()
        return DataLoader(
            TensorDataset(xt, yt),
            batch_size=cfg.train.batch_size,
            shuffle=shuffle,
            num_workers=cfg.train.num_workers,
        )

    train_loader = make_loader(train_mask, True)
    val_loader = make_loader(val_mask, False)
    print(f"train samples: {int(train_mask.sum())}, val samples: {int(val_mask.sum())}, "
          f"vocab: {NUM_MOVES}, device: {device}", file=sys.stderr)

    model = build_model(cfg.model).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    best_top1 = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / max(1, int(train_mask.sum()))

        model.eval()
        v_top1 = v_top3 = v_n = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                t1, t3 = _accuracy(logits, yb)
                bs = yb.size(0)
                v_top1 += t1 * bs
                v_top3 += t3 * bs
                v_n += bs
        top1 = v_top1 / max(1, v_n)
        top3 = v_top3 / max(1, v_n)
        print(f"epoch {epoch:3d}  train_loss {train_loss:.3f}  "
              f"val_top1 {top1:.3%}  val_top3 {top3:.3%}", file=sys.stderr)

        if top1 > best_top1:
            best_top1 = top1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.train.patience:
                print(f"early stopping at epoch {epoch}", file=sys.stderr)
                break

    out_path = config.model_path(user)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "model_cfg": vars(cfg.model),
            "num_moves": NUM_MOVES,
            "user": user,
            "val_top1": best_top1,
        },
        out_path,
    )
    print(f"Saved model to {out_path} (best val_top1 {best_top1:.3%})")
    return {"val_top1": best_top1, "model_path": str(out_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the imitation policy net.")
    parser.add_argument("--user", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    cfg = config.Config()
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr

    train(args.user, cfg, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
