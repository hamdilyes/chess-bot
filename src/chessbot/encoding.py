"""Board and move encoding.

This is the single most correctness-critical module. Two ideas:

1. **Side-to-move orientation.** We always present a position from the perspective of the
   player to move, as if they were White. When it is Black to move we use
   ``board.mirror()`` (python-chess mirrors vertically *and* swaps colors), giving an
   equivalent White-to-move position. The model is therefore color-agnostic and the
   player's White and Black games train the same weights.

   The matching transform for a move is a vertical square mirror (``sq ^ 56``), with the
   promotion piece unchanged. ``orient_move`` maps a real move into the White frame;
   ``deorient_move`` is its inverse (the mirror is an involution, so it is the same op).

2. **Fixed move vocabulary.** A deterministic superset of every legal "oriented" move:
   all queen-geometry and knight-geometry moves, plus pawn promotions on the 8th rank.
   The policy head outputs one logit per vocab entry; illegal moves are masked at
   inference. A superset (slightly larger than the canonical 1968) is fine: unused logits
   are simply never selected.
"""

from __future__ import annotations

import chess
import numpy as np

# --- Plane layout -----------------------------------------------------------------
# 0-5   : our pieces (P, N, B, R, Q, K) in the oriented (White-to-move) frame
# 6-11  : their pieces (P, N, B, R, Q, K)
# 12-15 : castling rights (our K-side, our Q-side, their K-side, their Q-side)
# 16    : en-passant target square
NUM_PLANES = 17

_PIECE_ORDER = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


def _is_queen_geometry(frm: int, to: int) -> bool:
    df = chess.square_file(to) - chess.square_file(frm)
    dr = chess.square_rank(to) - chess.square_rank(frm)
    return df == 0 or dr == 0 or abs(df) == abs(dr)


def _is_knight_geometry(frm: int, to: int) -> bool:
    df = abs(chess.square_file(to) - chess.square_file(frm))
    dr = abs(chess.square_rank(to) - chess.square_rank(frm))
    return (df, dr) in ((1, 2), (2, 1))


def _build_move_vocab() -> list[str]:
    moves: list[str] = []
    # Non-promotion moves: any queen- or knight-geometry from/to pair.
    for frm in range(64):
        for to in range(64):
            if frm == to:
                continue
            if _is_queen_geometry(frm, to) or _is_knight_geometry(frm, to):
                moves.append(chess.Move(frm, to).uci())
    # Promotions in the oriented frame: pawn from rank 7 (index 6) to rank 8 (index 7),
    # straight or diagonal, for each promotion piece.
    for from_file in range(8):
        frm = chess.square(from_file, 6)
        for to_file in (from_file - 1, from_file, from_file + 1):
            if 0 <= to_file < 8:
                to = chess.square(to_file, 7)
                for promo in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                    moves.append(chess.Move(frm, to, promotion=promo).uci())
    return moves


MOVE_VOCAB: list[str] = _build_move_vocab()
MOVE_TO_IDX: dict[str, int] = {uci: i for i, uci in enumerate(MOVE_VOCAB)}
NUM_MOVES: int = len(MOVE_VOCAB)


def mirror_move(move: chess.Move) -> chess.Move:
    """Vertically mirror a move (a1<->a8). Promotion piece is unchanged."""
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def orient_move(move: chess.Move, turn: chess.Color) -> chess.Move:
    """Map a real-board move into the oriented (White-to-move) frame."""
    return move if turn == chess.WHITE else mirror_move(move)


def deorient_move(move: chess.Move, turn: chess.Color) -> chess.Move:
    """Inverse of :func:`orient_move` (mirror is an involution)."""
    return move if turn == chess.WHITE else mirror_move(move)


def _oriented_board(board: chess.Board) -> chess.Board:
    """Return a White-to-move view of ``board`` (mirror if Black to move)."""
    return board if board.turn == chess.WHITE else board.mirror()


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a position as a ``(17, 8, 8)`` float32 tensor from the mover's view."""
    b = _oriented_board(board)
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for square, piece in b.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        base = 0 if piece.color == chess.WHITE else 6  # white == "us" after orientation
        plane = base + _PIECE_ORDER.index(piece.piece_type)
        planes[plane, rank, file] = 1.0

    if b.has_kingside_castling_rights(chess.WHITE):
        planes[12, :, :] = 1.0
    if b.has_queenside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if b.has_kingside_castling_rights(chess.BLACK):
        planes[14, :, :] = 1.0
    if b.has_queenside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0

    if b.ep_square is not None:
        rank = chess.square_rank(b.ep_square)
        file = chess.square_file(b.ep_square)
        planes[16, rank, file] = 1.0

    return planes


def move_to_index(move: chess.Move, turn: chess.Color) -> int:
    """Vocab index of a real-board ``move`` played by side ``turn``."""
    return MOVE_TO_IDX[orient_move(move, turn).uci()]


def legal_index_map(board: chess.Board) -> dict[int, chess.Move]:
    """Map each legal move's vocab index -> the real ``chess.Move`` for this board.

    Used to mask the policy to legal moves and to recover the move to actually play.
    """
    out: dict[int, chess.Move] = {}
    turn = board.turn
    for move in board.legal_moves:
        idx = MOVE_TO_IDX[orient_move(move, turn).uci()]
        out[idx] = move
    return out
