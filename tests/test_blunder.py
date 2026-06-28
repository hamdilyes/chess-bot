"""Tests for the shallow blunder detector on known positions."""

from __future__ import annotations

import chess

from chessbot import blunder


def test_hanging_queen_is_blunder():
    # After 1.e4 e5 2.Qh5 Nc6, the move 3.Qxe5?? hangs the queen to ...Nxe5.
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR w KQkq - 2 3")
    qxe5 = chess.Move.from_uci("h5e5")
    assert qxe5 in board.legal_moves
    assert blunder.is_one_move_blunder(board, qxe5)


def test_safe_developing_move_is_not_blunder():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR w KQkq - 2 3")
    nf3 = chess.Move.from_uci("g1f3")
    assert nf3 in board.legal_moves
    assert not blunder.is_one_move_blunder(board, nf3)


def test_winning_a_free_piece_is_not_blunder():
    # White to move with a free knight capture available (Black knight on e5 is undefended).
    board = chess.Board("rnbqkb1r/pppp1ppp/8/4n3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1")
    nxe5 = chess.Move.from_uci("f3e5")
    assert nxe5 in board.legal_moves
    assert not blunder.is_one_move_blunder(board, nxe5)


def test_starting_position_has_no_blunders():
    board = chess.Board()
    blunders, _ = blunder.blunder_set(board)
    assert blunders == set()
