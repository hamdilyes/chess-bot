"""Turn a player's PGN into training samples and measure their blunder rate.

For every position where the *target player* is to move we emit one sample:
``(encoded board, move index)``. We also record which game each sample came from so the
trainer can split train/val by game (no position leaks between the two). Finally we run the
shallow blunder detector over the player's own moves to estimate how often they blunder,
which calibrates the bot's blunder-imitation behavior.
"""

from __future__ import annotations

import argparse
import io
import json
import sys

import chess
import chess.pgn
import numpy as np
from tqdm import tqdm

from . import blunder, config
from .encoding import NUM_PLANES, encode_board, move_to_index


def _player_color(headers: chess.pgn.Headers, user: str) -> chess.Color | None:
    u = user.lower()
    if (headers.get("White") or "").lower() == u:
        return chess.WHITE
    if (headers.get("Black") or "").lower() == u:
        return chess.BLACK
    return None


def _is_standard(game: chess.pgn.Game) -> bool:
    variant = (game.headers.get("Variant") or "Standard").lower()
    # A FEN header means a non-standard start (e.g. Chess960 / from-position).
    has_setup = "FEN" in game.headers
    return variant in ("standard", "from position") and not has_setup


def _phase(fullmove_number: int) -> str:
    if fullmove_number <= 10:
        return "opening"
    if fullmove_number <= 30:
        return "middlegame"
    return "endgame"


def build_dataset(
    user: str,
    cfg: config.Config = config.DEFAULT,
    *,
    compute_blunders: bool = True,
    max_stat_positions: int | None = None,
) -> dict:
    """Parse ``data/{user}.pgn`` -> samples + stats. Returns a summary dict."""
    pgn_file = config.pgn_path(user)
    if not pgn_file.exists():
        raise FileNotFoundError(f"No PGN at {pgn_file}; run chessbot.download first.")

    xs: list[np.ndarray] = []
    ys: list[int] = []
    game_ids: list[int] = []

    # Blunder accounting (overall + per phase).
    blunder_counts = {"opening": 0, "middlegame": 0, "endgame": 0}
    move_counts = {"opening": 0, "middlegame": 0, "endgame": 0}
    stats_done = 0

    games_used = 0
    game_id = -1

    text = pgn_file.read_text(encoding="utf-8", errors="replace")
    stream = io.StringIO(text)

    pbar = tqdm(desc="games", unit="game")
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        pbar.update(1)

        if not _is_standard(game):
            continue
        color = _player_color(game.headers, user)
        if color is None:
            continue

        moves = list(game.mainline_moves())
        if len(moves) < cfg.data.min_plies:
            continue

        board = game.board()
        this_game_id = game_id + 1
        added_any = False

        for move in moves:
            if board.turn == color:
                xs.append(encode_board(board).astype(np.uint8))
                ys.append(move_to_index(move, board.turn))
                game_ids.append(this_game_id)
                added_any = True

                if compute_blunders and (
                    max_stat_positions is None or stats_done < max_stat_positions
                ):
                    phase = _phase(board.fullmove_number)
                    move_counts[phase] += 1
                    blunders, _ = blunder.blunder_set(
                        board,
                        cfg.blunder.material_threshold,
                        cfg.blunder.quiescence_depth,
                    )
                    if move in blunders:
                        blunder_counts[phase] += 1
                    stats_done += 1
            board.push(move)

        if added_any:
            game_id = this_game_id
            games_used += 1

    pbar.close()

    if not xs:
        raise RuntimeError(
            f"No samples for '{user}'. Check the username matches the PGN headers."
        )

    X = np.stack(xs).astype(np.uint8)  # (N, 17, 8, 8)
    y = np.asarray(ys, dtype=np.int32)
    gids = np.asarray(game_ids, dtype=np.int32)

    out_path = config.samples_path(user)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y, game_ids=gids, num_planes=NUM_PLANES)

    total_moves = sum(move_counts.values())
    total_blunders = sum(blunder_counts.values())
    overall_rate = (total_blunders / total_moves) if total_moves else 0.0
    phase_rates = {
        p: (blunder_counts[p] / move_counts[p]) if move_counts[p] else 0.0
        for p in move_counts
    }

    stats = {
        "user": user,
        "games_used": games_used,
        "samples": int(X.shape[0]),
        "blunder_positions_scored": stats_done,
        "blunder_rate": overall_rate,
        "blunder_rate_by_phase": phase_rates,
        "material_threshold": cfg.blunder.material_threshold,
        "quiescence_depth": cfg.blunder.quiescence_depth,
    }
    config.stats_path(user).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build training samples from a PGN.")
    parser.add_argument("--user", required=True)
    parser.add_argument(
        "--no-blunders",
        action="store_true",
        help="skip the (slower) blunder-rate measurement",
    )
    parser.add_argument(
        "--max-stat-positions",
        type=int,
        default=None,
        help="cap positions scored for the blunder rate (default: all)",
    )
    args = parser.parse_args(argv)

    stats = build_dataset(
        args.user,
        compute_blunders=not args.no_blunders,
        max_stat_positions=args.max_stat_positions,
    )
    print(f"Saved {stats['samples']} samples from {stats['games_used']} games "
          f"to {config.samples_path(args.user)}", file=sys.stderr)
    print(f"Player one-move-blunder rate: {stats['blunder_rate']:.3%} "
          f"(by phase: " + ", ".join(
              f"{p} {r:.2%}" for p, r in stats["blunder_rate_by_phase"].items()) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
