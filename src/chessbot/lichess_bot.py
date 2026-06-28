"""Play on lichess as a BOT account.

Streams incoming events with ``berserk``, auto-accepts standard challenges, and plays each
game with the trained :class:`~chessbot.engine.Engine`. One-time setup: the token's account
must be upgraded to a BOT account (``--upgrade``), which lichess only allows on an account
that has never played a game.

Usage:
    uv run python -m chessbot.lichess_bot --upgrade           # one time, on a fresh account
    uv run python -m chessbot.lichess_bot --user <player>     # run the bot
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

import berserk
import chess
from dotenv import load_dotenv

from .engine import Engine, load_engine


def _make_client(token: str) -> berserk.Client:
    return berserk.Client(session=berserk.TokenSession(token))


def _acceptable_challenge(challenge: dict) -> bool:
    variant = (challenge.get("variant") or {}).get("key", "standard")
    return variant == "standard"


def _board_from_state(initial_fen: str | None, moves: str) -> chess.Board:
    board = chess.Board() if not initial_fen or initial_fen == "startpos" else chess.Board(initial_fen)
    for uci in moves.split():
        board.push(chess.Move.from_uci(uci))
    return board


class LichessBot:
    def __init__(self, client: berserk.Client, engine: Engine):
        self.client = client
        self.engine = engine
        self.my_id = client.account.get()["id"].lower()
        print(f"Logged in as bot '{self.my_id}'", file=sys.stderr)

    def run(self) -> None:
        print("Waiting for challenges. Challenge this bot from your main account.", file=sys.stderr)
        for event in self.client.bots.stream_incoming_events():
            etype = event.get("type")
            if etype == "challenge":
                self._on_challenge(event["challenge"])
            elif etype == "gameStart":
                game = event["game"]
                game_id = game.get("gameId") or game.get("id")
                threading.Thread(target=self._play_game, args=(game_id,), daemon=True).start()

    def _on_challenge(self, challenge: dict) -> None:
        cid = challenge["id"]
        if challenge.get("challenger", {}).get("id", "").lower() == self.my_id:
            return  # ignore our own outgoing challenges
        try:
            if _acceptable_challenge(challenge):
                self.client.bots.accept_challenge(cid)
                print(f"Accepted challenge {cid}", file=sys.stderr)
            else:
                self.client.bots.decline_challenge(cid)
                print(f"Declined non-standard challenge {cid}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - lichess may have already expired it
            print(f"Challenge {cid} handling failed: {exc}", file=sys.stderr)

    def _play_game(self, game_id: str) -> None:
        try:
            stream = self.client.bots.stream_game_state(game_id)
            full = next(stream)
            my_color = chess.WHITE if full["white"].get("id") == self.my_id else chess.BLACK
            initial_fen = full.get("initialFen", "startpos")
            print(f"Game {game_id}: playing as {'white' if my_color else 'black'}", file=sys.stderr)

            self._maybe_move(game_id, initial_fen, full["state"], my_color)
            for event in stream:
                if event.get("type") == "gameState":
                    self._maybe_move(game_id, initial_fen, event, my_color)
        except Exception as exc:  # noqa: BLE001
            print(f"Game {game_id} loop error: {exc}", file=sys.stderr)

    def _maybe_move(self, game_id: str, initial_fen: str, state: dict, my_color: chess.Color) -> None:
        if state.get("status", "started") != "started":
            return  # game finished/aborted
        board = _board_from_state(initial_fen, state.get("moves", ""))
        if board.is_game_over() or board.turn != my_color:
            return
        try:
            move = self.engine.select_move(board)
            self.client.bots.make_move(game_id, move.uci())
        except Exception as exc:  # noqa: BLE001
            print(f"Game {game_id} move error ({exc}); resigning.", file=sys.stderr)
            try:
                self.client.bots.resign_game(game_id)
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the chess bot on a lichess BOT account.")
    parser.add_argument("--user", help="trained player model to play as (the imitated player)")
    parser.add_argument("--token", default=os.environ.get("LICHESS_TOKEN"))
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="upgrade this token's account to a BOT account (one-time, fresh account only)",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    if not args.token:
        print("No token. Set LICHESS_TOKEN in .env or pass --token.", file=sys.stderr)
        return 2

    client = _make_client(args.token)

    if args.upgrade:
        client.account.upgrade_to_bot()
        print("Account upgraded to BOT.")
        return 0

    if not args.user:
        print("Pass --user <player> (the trained model to play as).", file=sys.stderr)
        return 2

    engine = load_engine(args.user, device=args.device)
    LichessBot(client, engine).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
