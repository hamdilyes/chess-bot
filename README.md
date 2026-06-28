# chess-bot

A chess bot that learns to play like **one specific lichess player**. It starts knowing
only the rules (no pretrained weights), downloads that player's games from lichess, trains
a neural network to imitate their moves (behavioral cloning), and plays full games as a
real **lichess BOT account** so you can challenge it on lichess.org.

It also reproduces the player's *blunder habits*: a shallow material search flags one-move
blunders, and the bot keeps or avoids them at a rate calibrated from how often the player
actually blunders.

## How it works

| Stage | Module | What it does |
|-------|--------|--------------|
| Download | `download.py` | Stream the player's games to `data/<user>.pgn` |
| Dataset | `dataset.py` | PGN → `(position, move)` samples + measure the player's blunder rate |
| Model | `model.py` | Small AlphaZero-style residual CNN, policy head over a fixed ~1880-move vocabulary |
| Train | `train.py` | Behavioral cloning; reports validation top-1/top-3 move-match accuracy |
| Blunder | `blunder.py` | Engine-free one-move-blunder detector (material quiescence search) |
| Engine | `engine.py` | Policy net + blunder-aware move selection |
| Play | `lichess_bot.py` | Run as a lichess BOT account; `uci.py` is a UCI shim for offline play |

Positions are always encoded from the side-to-move's perspective (Black-to-move boards are
mirrored to look White-to-move), so the player's White and Black games train the same
weights. See `encoding.py` — this is the most correctness-critical piece and is covered by
`tests/test_encoding.py`.

## Setup

```bash
uv sync --extra dev
```

Put your lichess BOT token in `.env` (gitignored):

```
LICHESS_TOKEN=lip_xxxxxxxxxxxx
```

## Run the pipeline

```bash
# 1. Download a player's games (the player to imitate)
uv run python -m chessbot.download --user <player> --max 2000

# 2. Build training samples + measure their blunder rate
uv run python -m chessbot.dataset --user <player>

# 3. Train the imitation policy
uv run python -m chessbot.train --user <player>

# 4a. Play it locally in any UCI GUI / against Stockfish
uv run python -m chessbot.uci --user <player>

# 4b. Or run it live on lichess (see below)
uv run python -m chessbot.lichess_bot --user <player>
```

## Lichess BOT account (one-time setup)

A BOT account must be a **fresh account that has never played a game**. To convert it:

```bash
uv run python -m chessbot.lichess_bot --upgrade
```

This is **irreversible** (the account can no longer play as a human). Then run the bot and
challenge it from your main account:

```bash
uv run python -m chessbot.lichess_bot --user <player>
```

## Tuning

All thresholds live in `config.py`:

- `BlunderConfig.p_keep_high` / `p_keep_low` — keep-blunder probability for frequent vs rare
  blunderers (defaults 0.65 / 0.10).
- `BlunderConfig.frequent_rate_threshold` — blunder rate above which the player is treated
  as a frequent blunderer.
- `BlunderConfig.material_threshold` / `quiescence_depth` — sensitivity of the detector.
- `BlunderConfig.top_k_sample` — set > 1 to sample among the top-k policy moves for variety.

## Honest expectations

One player has limited data (a few hundred to a few thousand games), so the bot is a
believable *imitation* of that player's tendencies, not a strong engine. Validation top-1
move-match typically lands around 30–45% with enough games (versus a ~3% random-legal
baseline). Strength scales with the player's game count and rating.

## Tests

```bash
uv run pytest -q
```
