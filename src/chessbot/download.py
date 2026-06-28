"""Download a lichess player's games as PGN.

Streams from the public lichess games export endpoint:
    GET https://lichess.org/api/games/user/{username}

No auth is required for public games, but supplying a token (``--token`` or the
``LICHESS_TOKEN`` env var / ``.env``) raises the rate limit and lets you pull more games.
"""

from __future__ import annotations

import argparse
import os
import sys

import requests
from dotenv import load_dotenv

from . import config

GAMES_URL = "https://lichess.org/api/games/user/{user}"


def download_games(
    user: str,
    *,
    max_games: int | None = 2000,
    perfs: tuple[str, ...] = config.DEFAULT.data.keep_perfs,
    rated: bool | None = None,
    token: str | None = None,
) -> int:
    """Stream a user's games to ``data/{user}.pgn``. Returns the number of games saved."""
    params: dict[str, str] = {
        "clocks": "false",
        "evals": "false",
        "opening": "false",
        "moves": "true",
    }
    if max_games is not None:
        params["max"] = str(max_games)
    if perfs:
        params["perfType"] = ",".join(perfs)
    if rated is not None:
        params["rated"] = "true" if rated else "false"

    headers = {"Accept": "application/x-chess-pgn"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    out_path = config.pgn_path(user)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    games = 0
    url = GAMES_URL.format(user=user)
    with requests.get(url, params=params, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with out_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                fh.write(chunk)
                # Each game's PGN starts with an Event tag; count them as a progress hint.
                games += chunk.count(b"[Event ")
    return games


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Download a lichess player's games as PGN.")
    parser.add_argument("--user", required=True, help="lichess username to learn from")
    parser.add_argument("--max", type=int, default=2000, help="max games to download")
    parser.add_argument(
        "--perfs",
        default=",".join(config.DEFAULT.data.keep_perfs),
        help="comma-separated lichess perf types (e.g. blitz,rapid,classical)",
    )
    parser.add_argument(
        "--rated",
        choices=["true", "false", "any"],
        default="any",
        help="filter by rated/casual games",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("LICHESS_TOKEN"),
        help="lichess API token (defaults to LICHESS_TOKEN env var / .env)",
    )
    args = parser.parse_args(argv)

    perfs = tuple(p for p in args.perfs.split(",") if p)
    rated = None if args.rated == "any" else (args.rated == "true")

    print(f"Downloading up to {args.max} games for '{args.user}' ...", file=sys.stderr)
    count = download_games(
        args.user, max_games=args.max, perfs=perfs, rated=rated, token=args.token
    )
    print(f"Saved {count} games to {config.pgn_path(args.user)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
