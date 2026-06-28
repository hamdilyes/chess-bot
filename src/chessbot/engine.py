"""The play-time engine: policy net + blunder-aware move selection.

``Engine.select_move(board)`` is the single entry point used by the lichess loop and the
UCI shim. It runs the imitation policy, then applies the blunder logic:

* If the policy's chosen move is **not** a one-move blunder, play it.
* If it **is** a blunder, keep it with probability ``p_keep`` (imitating the player) or
  otherwise switch to the best non-blunder move by policy ("saw it, avoided it").

``p_keep`` is calibrated from the player's measured blunder rate: frequent blunderers keep
their blunders most of the time (~0.65), rare blunderers avoid them most of the time.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import chess
import numpy as np
import torch

from . import config
from .blunder import blunder_set
from .config import BlunderConfig, ModelConfig
from .encoding import encode_board, legal_index_map
from .model import build_model


class Engine:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        blunder_rate: float = 0.0,
        blunder_cfg: BlunderConfig | None = None,
        device: str = "cpu",
        seed: int | None = None,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.blunder_rate = blunder_rate
        self.bcfg = blunder_cfg or BlunderConfig()
        self.rng = random.Random(seed)

    # --- construction ------------------------------------------------------------
    @classmethod
    def from_user(
        cls, user: str, *, device: str = "cpu", seed: int | None = None
    ) -> "Engine":
        ckpt = torch.load(config.model_path(user), map_location=device, weights_only=False)
        model = build_model(ModelConfig(**ckpt["model_cfg"]))
        model.load_state_dict(ckpt["model_state"])

        blunder_rate = 0.0
        stats_path = config.stats_path(user)
        if stats_path.exists():
            blunder_rate = json.loads(stats_path.read_text()).get("blunder_rate", 0.0)
        return cls(model, blunder_rate=blunder_rate, device=device, seed=seed)

    # --- policy ------------------------------------------------------------------
    def policy_probs(self, board: chess.Board) -> dict[chess.Move, float]:
        """Legal-masked softmax over the policy logits -> {move: probability}."""
        x = torch.from_numpy(encode_board(board)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x).squeeze(0).cpu().numpy()

        idx_map = legal_index_map(board)  # {vocab_idx: real move}
        idxs = np.fromiter(idx_map.keys(), dtype=np.int64)
        legal_logits = logits[idxs]
        legal_logits -= legal_logits.max()  # numerical stability
        probs = np.exp(legal_logits)
        probs /= probs.sum()
        return {idx_map[int(i)]: float(p) for i, p in zip(idxs, probs)}

    # --- blunder calibration -----------------------------------------------------
    def _p_keep(self) -> float:
        if self.blunder_rate >= self.bcfg.frequent_rate_threshold:
            return self.bcfg.p_keep_high
        return self.bcfg.p_keep_low

    # --- main entry point --------------------------------------------------------
    def select_move(self, board: chess.Board) -> chess.Move:
        probs = self.policy_probs(board)
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)

        # Candidate move: argmax, or sample among the top-k by probability for variety.
        k = max(1, self.bcfg.top_k_sample)
        if k == 1:
            candidate = ranked[0][0]
        else:
            top = ranked[:k]
            weights = [p for _, p in top]
            candidate = self.rng.choices([m for m, _ in top], weights=weights, k=1)[0]

        blunders, _ = blunder_set(
            board, self.bcfg.material_threshold, self.bcfg.quiescence_depth
        )
        if candidate not in blunders:
            return candidate

        # The policy wants to blunder. Imitate it, or avoid it.
        if self.rng.random() < self._p_keep():
            return candidate
        # Avoid: best non-blunder move by policy; fall back to candidate if everything hangs.
        for move, _ in ranked:
            if move not in blunders:
                return move
        return candidate


def load_engine(user: str, **kwargs) -> Engine:
    if not Path(config.model_path(user)).exists():
        raise FileNotFoundError(
            f"No trained model for '{user}'. Run download -> dataset -> train first."
        )
    return Engine.from_user(user, **kwargs)
