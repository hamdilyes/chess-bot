"""Minimal UCI shim around the engine.

Lets you play the bot locally in any chess GUI or pit it against Stockfish to gauge real
strength. Only the subset of UCI needed to play a game is implemented.

    uv run python -m chessbot.uci --user <player>
"""

from __future__ import annotations

import argparse
import sys

import chess

from .engine import load_engine


def _apply_position(line: str) -> chess.Board:
    tokens = line.split()
    board = chess.Board()
    idx = 1
    if idx < len(tokens) and tokens[idx] == "startpos":
        idx += 1
    elif idx < len(tokens) and tokens[idx] == "fen":
        fen = " ".join(tokens[idx + 1 : idx + 7])
        board = chess.Board(fen)
        idx += 7
    if idx < len(tokens) and tokens[idx] == "moves":
        for uci in tokens[idx + 1 :]:
            board.push(chess.Move.from_uci(uci))
    return board


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="UCI shim for the chess bot.")
    parser.add_argument("--user", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    engine = load_engine(args.user, device=args.device)
    board = chess.Board()

    for line in sys.stdin:
        line = line.lstrip("﻿").strip()
        if line == "uci":
            print(f"id name chessbot-{args.user}")
            print("id author chessbot")
            print("uciok")
        elif line == "isready":
            print("readyok")
        elif line == "ucinewgame":
            board = chess.Board()
        elif line.startswith("position"):
            board = _apply_position(line)
        elif line.startswith("go"):
            move = engine.select_move(board)
            print(f"bestmove {move.uci()}")
        elif line == "quit":
            break
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
