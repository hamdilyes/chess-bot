"""Central configuration: paths and tunable thresholds.

Everything tunable lives here so the rest of the code reads from one place. Paths are
resolved relative to the repository root so the CLIs work regardless of the working dir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Repo root = three levels up from this file: src/chessbot/config.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"


def pgn_path(user: str) -> Path:
    return DATA_DIR / f"{user.lower()}.pgn"


def samples_path(user: str) -> Path:
    return DATA_DIR / f"{user.lower()}_samples.npz"


def stats_path(user: str) -> Path:
    return DATA_DIR / f"{user.lower()}_stats.json"


def model_path(user: str) -> Path:
    return MODELS_DIR / f"{user.lower()}.pt"


@dataclass
class DataConfig:
    """Filters applied while turning PGN into training samples."""

    # Time-control families to keep (lichess `perfType`). Empty = keep all standard.
    keep_perfs: tuple[str, ...] = ("blitz", "rapid", "classical")
    min_plies: int = 8  # skip ultra-short games / abandoned games
    val_fraction: float = 0.1  # held-out games for validation (split by game)


@dataclass
class ModelConfig:
    in_planes: int = 17
    channels: int = 96
    blocks: int = 6


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5  # early-stop on val top-1 move-match accuracy
    num_workers: int = 0  # Windows-friendly default


@dataclass
class BlunderConfig:
    """Thresholds for the shallow blunder detector and blunder-aware selection."""

    # A move is a "one-move blunder" if it concedes at least this much material
    # (in pawns) versus the best safe alternative. 2.0 ≈ a minor piece.
    material_threshold: float = 2.0
    quiescence_depth: int = 2
    # If the player's measured one-move-blunder rate exceeds this, treat them as a
    # "frequent blunderer" and keep blunders with `p_keep_high`.
    frequent_rate_threshold: float = 0.04
    p_keep_high: float = 0.65  # keep-blunder probability for frequent blunderers
    p_keep_low: float = 0.10   # keep-blunder probability for rare blunderers
    top_k_sample: int = 1      # 1 = always take argmax policy move; >1 = sample top-k


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    blunder: BlunderConfig = field(default_factory=BlunderConfig)


DEFAULT = Config()
