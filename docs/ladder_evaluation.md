# Ladder Evaluation

Use ladder runs as the project-level strength signal once a change has passed cheaper offline checks.  Raw Showdown Elo gain is useful, but it is not enough by itself because matchmaking, time of day, early-rating volatility, and opponent quality can dominate short runs.

The ladder runner now writes one JSONL row per finished battle with opponent-adjusted metrics.

## Run A Ladder Canary

Set credentials outside git:

```bash
export PS_USERNAME="your_bot_account"
export PS_PASSWORD="your_password"
```

Example Oranguru ladder run:

```bash
PYTHON_BIN="${PYTHON_BIN:-$(pwd)/venv/bin/python}"
RUN_TAG="oranguru_static8_800ms_$(date +%Y%m%d_%H%M%S)"

ORANGURU_BOT_VERSION="$RUN_TAG" \
ORANGURU_SEARCH_MS=800 \
ORANGURU_SAMPLE_STATES=8 \
ORANGURU_SAMPLE_STATES_MAX=8 \
ORANGURU_DYNAMIC_SAMPLING=0 \
ORANGURU_HEURISTIC_BLEND=0.0 \
ORANGURU_MIN_HEURISTIC_BLEND=0.0 \
ORANGURU_TACTICAL_RERANKS=0 \
ORANGURU_FINISH_BLOW_GUARD=1 \
ORANGURU_SETUP_WINDOW=0 \
ORANGURU_RECOVERY_WINDOW=0 \
ORANGURU_PROGRESS_WINDOW=0 \
ORANGURU_SWITCH_GUARD=0 \
"$PYTHON_BIN" evaluation/ladder_rulebot.py \
  --player oranguru_engine \
  --format gen9randombattle \
  --battles 100 \
  --max-concurrent 1 \
  --progress-every 10 \
  --auto-reconnect \
  --rejoin-active \
  --ladder-log "logs/ladder/${RUN_TAG}.jsonl" \
  --snapshot-log "logs/ladder/${RUN_TAG}.snapshots.jsonl"
```

## Summarize Results

```bash
"$PYTHON_BIN" evaluation/summarize_ladder_metrics.py \
  logs/ladder/oranguru_static8_800ms_*.jsonl \
  --by-version \
  --by-account \
  --json-out logs/ladder/ladder_summary.json
```

## Logged Fields

Each ladder metrics row includes:

- `bot_version`: label from `--bot-version`, `ORANGURU_BOT_VERSION`, or git commit
- `account`, `player_type`, `format`, `battle_tag`, `opponent_username`
- `result`, `score`, `turns`, `remaining`, `opp_remaining`, `forfeit`
- `player_rating_pre`, `player_rating_post`, `player_rating_delta`
- `opponent_rating_pre`, `opponent_rating_post`, `opponent_rating_delta`
- `expected_score`: Elo expectation from pre-ratings
- `rating_residual`: actual score minus expected score
- `action_counts`, `decision_count`, `decision_ms_avg`, `decision_ms_max`

## Preferred Metric

Rank versions by rating-adjusted residual first, then use raw win rate and final rating as supporting context:

```text
actual_score = win ? 1 : loss ? 0 : 0.5
expected_score = 1 / (1 + 10 ** ((opp_rating_pre - our_rating_pre) / 400))
rating_residual = actual_score - expected_score
```

A positive average residual means the bot beat expectation for the opponents it was matched against.

## Evaluation Stack

Use these layers in order:

1. Offline FP smoke tests to reject obvious regressions.
2. Trace/teacher benchmarks for cheap decision-quality checks.
3. Ladder canaries for real-world confirmation.

Avoid deciding from a single short ladder run. Prefer repeated windows with a fixed bot version label and compare residuals across similar time periods.
