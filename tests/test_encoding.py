"""Correctness tests for the encoding module.

The flip round-trip and vocab-coverage tests are the safety net for the whole project:
if orientation is wrong the bot plays mirror-image nonsense, and if the vocab misses a
legal move training/inference will crash.
"""

from __future__ import annotations

import random

import chess
import numpy as np

from chessbot import encoding as enc


def _random_positions(n: int, max_plies: int = 40, seed: int = 0) -> list[chess.Board]:
    rng = random.Random(seed)
    boards: list[chess.Board] = []
    for _ in range(n):
        board = chess.Board()
        plies = rng.randint(0, max_plies)
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        boards.append(board)
    return boards


def test_vocab_covers_all_legal_moves():
    """Every legal move in many random positions must map into the vocabulary."""
    for board in _random_positions(400, seed=1):
        for move in board.legal_moves:
            oriented = enc.orient_move(move, board.turn).uci()
            assert oriented in enc.MOVE_TO_IDX, f"missing {oriented} from {board.fen()}"


def test_orient_deorient_round_trip():
    """deorient(orient(move)) == move for both colors."""
    for board in _random_positions(400, seed=2):
        for move in board.legal_moves:
            oriented = enc.orient_move(move, board.turn)
            back = enc.deorient_move(oriented, board.turn)
            assert back == move


def test_legal_index_map_recovers_moves():
    """The index map must round-trip indices back to the exact legal moves."""
    for board in _random_positions(200, seed=3):
        idx_map = enc.legal_index_map(board)
        assert len(idx_map) == board.legal_moves.count()
        for move in board.legal_moves:
            idx = enc.move_to_index(move, board.turn)
            assert idx_map[idx] == move


def test_encode_shape_and_orientation():
    """Encoding is always (17,8,8); a position and its color-mirror encode identically."""
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    planes = enc.encode_board(board)
    assert planes.shape == (enc.NUM_PLANES, 8, 8)
    assert planes.dtype == np.float32

    # Black-to-move position vs its mirror (White to move) must encode the same, because
    # we always view from the side to move.
    mirrored = board.mirror()
    assert np.array_equal(enc.encode_board(board), enc.encode_board(mirrored))


def test_starting_position_piece_counts():
    planes = enc.encode_board(chess.Board())
    # 8 pawns + 2N + 2B + 2R + 1Q + 1K per side = 16 each.
    assert planes[0:6].sum() == 16  # our pieces
    assert planes[6:12].sum() == 16  # their pieces
    # All castling rights present in the start position.
    for p in (12, 13, 14, 15):
        assert planes[p].sum() == 64
    assert planes[16].sum() == 0  # no en-passant square
