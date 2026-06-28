"""Shallow, engine-free blunder detection.

A *one-move blunder* here means: a move that concedes material an immediate tactical
sequence can win, when a clearly safer legal alternative existed. We evaluate each move
with a tiny negamax + capture-only quiescence search using a pure material eval. This is
cheap (no external engine, no neural net) and catches the everyday "hung a piece" mistakes
that dominate amateur play.

Used in two places:
* ``dataset`` measures how often the target player makes these blunders.
* ``engine`` decides whether to imitate or avoid a blunder the policy proposes.
"""

from __future__ import annotations

import math

import chess

# Material values in pawns. King is effectively infinite but never captured.
PIECE_VALUES: dict[chess.PieceType, float] = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

_MATE = 1000.0


def _material_white_minus_black(board: chess.Board) -> float:
    total = 0.0
    for piece in board.piece_map().values():
        v = PIECE_VALUES[piece.piece_type]
        total += v if piece.color == chess.WHITE else -v
    return total


def _eval_side_to_move(board: chess.Board) -> float:
    """Material from the perspective of the side to move (positive = good for mover)."""
    bal = _material_white_minus_black(board)
    return bal if board.turn == chess.WHITE else -bal


def _quiesce(board: chess.Board, alpha: float, beta: float, depth: int) -> float:
    """Negamax quiescence. Searches captures (and all evasions when in check)."""
    if board.is_checkmate():
        return -_MATE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    in_check = board.is_check()
    stand_pat = _eval_side_to_move(board)
    if depth == 0:
        return stand_pat

    if not in_check:
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

    # When in check we must consider every evasion; otherwise only captures matter for a
    # material quiescence.
    moves = (
        list(board.legal_moves)
        if in_check
        else [m for m in board.legal_moves if board.is_capture(m)]
    )
    for move in moves:
        board.push(move)
        score = -_quiesce(board, -beta, -alpha, depth - 1)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def move_values(board: chess.Board, depth: int = 2) -> dict[chess.Move, float]:
    """Material value (from the mover's perspective) reachable after each legal move."""
    values: dict[chess.Move, float] = {}
    for move in board.legal_moves:
        board.push(move)
        # After our move the opponent is to move; negate their best outcome.
        values[move] = -_quiesce(board, -math.inf, math.inf, depth)
        board.pop()
    return values


def blunder_set(
    board: chess.Board, threshold: float = 2.0, depth: int = 2
) -> tuple[set[chess.Move], dict[chess.Move, float]]:
    """Return (blunder moves, all move values).

    A move is a blunder if its value is at least ``threshold`` pawns worse than the best
    available move (i.e. a clearly safer alternative existed).
    """
    values = move_values(board, depth)
    if not values:
        return set(), values
    best = max(values.values())
    blunders = {m for m, v in values.items() if best - v >= threshold}
    return blunders, values


def is_one_move_blunder(
    board: chess.Board, move: chess.Move, threshold: float = 2.0, depth: int = 2
) -> bool:
    """True if ``move`` is a one-move blunder in ``board``."""
    blunders, _ = blunder_set(board, threshold, depth)
    return move in blunders
