# Oranguru MCTS

Oranguru MCTS is a Pokemon Showdown battle bot built around a Monte Carlo Tree Search engine for `gen9randombattle`.

The repository serves two purposes:
- a competitive search bot centered on `src/players/oranguru_engine.py`
- a research workspace for training helper models that support search without replacing it

The main production path is still the search engine. Learned components exist, but they are experimental unless an evaluation result has clearly promoted them.

## What is in the repo

```text
oranguru_mcts/
├── src/
│   ├── players/        # main engine, heuristic bot, older search player, RL player
│   ├── models/         # actor-critic and search-assist models
│   └── utils/          # feature building, damage, belief, and server helpers
├── evaluation/         # offline Foul Play evals, ladder runners, diagnostics, block runner
├── training/           # replay prep, training scripts, dataset builders
├── tests/              # focused engine and heuristic tests
├── data/               # static game data plus local-only generated artifacts
├── checkpoints/        # local model checkpoints (ignored by git)
├── logs/               # local eval logs and search traces (ignored by git)
├── third_party/        # vendored dependencies, notably Foul Play
├── backups/            # frozen reference snapshots and local safety backups
└── requirements.txt
```

## Main components

### Players
- `src/players/oranguru_engine.py`: main MCTS player and current evaluation target
- `src/players/rule_bot.py`: heuristic baseline and fallback logic
- `src/players/Oranguru.py`: older lightweight search player kept for reference
- `src/players/rl_player.py`: offline RL / imitation policy player

### Evaluation
- `evaluation/eval_vs_foulplay.py`: primary offline benchmark harness
- `evaluation/run_block_eval.py`: repeated-block runner for more trustworthy comparisons
- `evaluation/ladder_rulebot.py`: ladder play and trajectory collection
- `evaluation/run_ladder_triplet.py`: multi-account ladder runner

### Replay reporting API
- `src/api/oranguru_mcts_api.py`: dict-in/dict-out API that accepts a replay-public game state and returns the top MCTS move plus policy reasoning
- `src/replay/replay_report.py`: replay URL/local JSON parser that feeds reconstructed turns through the MCTS API sequentially
- `src/web/replay_report_server.py`: zero-dependency self-hosted form for "paste URL, get report"

### Training and data prep
- `training/train_search_prior_value.py`: trainer for search-assist prior/value models
- `training/prepare_search_assist_from_jsonl.py`: converts live search traces into pickled datasets
- `training/prepare_switch_prior_dataset.py`: switch-only pruning dataset builder
- `training/prepare_passive_break_dataset.py`: passive-line dataset builder
- `training/prepare_tera_pruner_dataset.py`: tera-only pruning dataset builder
- `training/train_sequence_bc.py` and `training/finetune_offline_rl.py`: older BC / offline RL pipeline
- `training/split_trajectory_pickle.py`: utility for splitting trajectory pickles into train/validation sets

## Practical status

Current working assumptions:
- `oranguru_engine` is the primary player under test
- 200-battle runs are smoke tests only
- 500-battle repeated blocks are the minimum bar for believing a change
- MCTS should not be replaced by a broad neural policy
- neural work should support search through targeted gates, priors, pruning, or calibration
- broad neural action steering is not trusted unless an eval clearly promotes it
- targeted helpers are more promising than general learned move selection

Current Foul Play direction:
- keep world ranker on
- keep passive breaker on
- keep switch guard on
- keep conservative setup/recovery/progress tactical windows
- disable the speculative `take_risk_attack` switch-guard branch by default
- use rerank trace attribution before making further behavior changes

## Setup

### 1. Create the environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r third_party/foul-play/requirements.txt
```

### 2. Install `poke-engine`

The engine and Foul Play both depend on `poke-engine`. The exact build can vary by platform, but the Gen 9 build used in this repo is typically installed like this:

```bash
source venv/bin/activate
pip install -v --force-reinstall --no-cache-dir poke-engine \
  --config-settings="build-args=--features poke-engine/gen9 --no-default-features"
```

## Offline evaluation

The standard benchmark is `evaluation/eval_vs_foulplay.py` against a local Showdown websocket and a local Foul Play process.

Minimal shape:

```bash
source venv/bin/activate
venv/bin/python evaluation/eval_vs_foulplay.py \
  --player oranguru_engine \
  --battles 200 \
  --format gen9randombattle \
  --foulplay-python venv/bin/python \
  --foulplay-username fp_local \
  --foulplay-password foulplay_pass \
  --ws-uri ws://127.0.0.1:8000/showdown/websocket
```

For serious comparisons, prefer `evaluation/run_block_eval.py` and repeated 500-battle blocks instead of one-off runs.

## Search traces and rerank review

Search traces are the main debugging artifact for tactical behavior work. Enable them in eval runs with:

```bash
export ORANGURU_SEARCH_TRACE=1
export ORANGURU_SEARCH_TRACE_OUT="logs/search_traces/current/<run_tag>.jsonl"
export ORANGURU_SEARCH_TRACE_TAG="<run_tag>"
```

Summarize decision paths and rerank attribution:

```bash
TRACE_OUT="logs/search_traces/current/<run_tag>.jsonl"
TRACE_DIAG_JSON="logs/replay_audit/<run_tag>.trace_decision_diag.json"

venv/bin/python evaluation/summarize_trace_decisions.py \
  "$TRACE_OUT" \
  --limit 50 \
  --json-out "$TRACE_DIAG_JSON"
```

Mine tactical review buckets:

```bash
TRACE_OUT="logs/search_traces/current/<run_tag>.jsonl"
RUN_TAG="$(basename "$TRACE_OUT" .jsonl)"
ISSUES_JSON="logs/replay_audit/${RUN_TAG}.bad_decisions.json"
REVIEW_JSON="logs/replay_audit/${RUN_TAG}.review_pack.json"

venv/bin/python evaluation/mine_bad_decisions.py --input "$TRACE_OUT" --summary-out "$ISSUES_JSON"
venv/bin/python evaluation/build_decision_review_pack.py --issues-json "$ISSUES_JSON" --output "$REVIEW_JSON" --limit 300

venv/bin/python evaluation/print_decision_review_pack.py --input "$REVIEW_JSON" --limit 40 --category ignored_safe_recovery
venv/bin/python evaluation/print_decision_review_pack.py --input "$REVIEW_JSON" --limit 40 --category underused_setup_window
venv/bin/python evaluation/print_decision_review_pack.py --input "$REVIEW_JSON" --limit 40 --category failed_to_progress_when_behind
venv/bin/python evaluation/print_decision_review_pack.py --input "$REVIEW_JSON" --limit 40 --category over_attacked_into_bad_trade
venv/bin/python evaluation/print_decision_review_pack.py --input "$REVIEW_JSON" --limit 40 --category over_switched_negative_matchup
venv/bin/python evaluation/print_decision_review_pack.py --input "$REVIEW_JSON" --limit 40 --category missed_ko
```

## Rerank gate neural hook

The rerank gate is a neural-hook scaffold for supporting MCTS. It is not a replacement policy. The hook can learn when to permit tactical reranks that displace the MCTS top action.

Default behavior is unchanged:

```bash
ORANGURU_RERANK_GATE=0
```

Build a weak-supervised rerank-gate dataset from accepted reranks:

```bash
RERANK_GATE_TAG="rerank_gate_$(date +%Y%m%d_%H%M%S)"
RERANK_GATE_JSONL="logs/replay_audit/${RERANK_GATE_TAG}.jsonl"
RERANK_GATE_SUMMARY="logs/replay_audit/${RERANK_GATE_TAG}.summary.json"

venv/bin/python evaluation/build_rerank_gate_dataset.py \
  "logs/search_traces/current/*conservative*_accept.jsonl" \
  "logs/search_traces/current/*switch_guard_risk_off_accept.jsonl" \
  --output "$RERANK_GATE_JSONL" \
  --summary-out "$RERANK_GATE_SUMMARY"

wc -l "$RERANK_GATE_JSONL"
cat "$RERANK_GATE_SUMMARY"
```

Build a focused dataset for the highest-volume sources:

```bash
venv/bin/python evaluation/build_rerank_gate_dataset.py \
  "logs/search_traces/current/*conservative*_accept.jsonl" \
  "logs/search_traces/current/*switch_guard_risk_off_accept.jsonl" \
  --source setup_window:take_setup \
  --source recovery_window:take_recovery \
  --source switch_guard:take_live_attack \
  --output "logs/replay_audit/${RERANK_GATE_TAG}.core_sources.jsonl" \
  --summary-out "logs/replay_audit/${RERANK_GATE_TAG}.core_sources.summary.json"
```

Runtime gate knobs:

```bash
export ORANGURU_RERANK_GATE=1
export ORANGURU_RERANK_GATE_MODEL="checkpoints/rl/rerank_gate.json"
export ORANGURU_RERANK_GATE_THRESHOLD=0.50
export ORANGURU_RERANK_GATE_FAIL_OPEN=1
```

Do not enable the runtime gate for serious evals until a calibrated `checkpoints/rl/rerank_gate.json` exists and has been smoke-tested.

## Replay report service

The replay report path is designed to run on the existing server with no new infrastructure service. It uses Python stdlib HTTP serving and the repo's existing `poke-engine` / Foul Play dependencies.

Start the form:

```bash
source venv/bin/activate
venv/bin/python -m src.web.replay_report_server --host 0.0.0.0 --port 8765
```

Then open `http://<server>:8765`, paste a public Pokemon Showdown replay URL, and run the report.

CLI equivalent:

```bash
source venv/bin/activate
venv/bin/python -m src.replay.replay_report \
  https://replay.pokemonshowdown.com/gen9randombattle-2390494148 \
  --player both \
  --search-ms 200 \
  --num-battles 4 \
  --max-decisions 24 \
  --output logs/replay_reports/report.json
```

Direct API shape:

```python
from src.api.oranguru_mcts_api import MCTSApiConfig, analyze_game_state

result = analyze_game_state(
    game_state_row,
    config=MCTSApiConfig(search_ms=200, num_battles=4),
)
print(result["top_move"], result["reasoning"]["summary"])
```

## Ladder accounts

Do not commit real ladder credentials.

This repo tracks `data/ladder_accounts.example.txt` only. To use ladder scripts locally:

```bash
cp data/ladder_accounts.example.txt data/ladder_accounts.txt
```

Then edit `data/ladder_accounts.txt` with your own accounts. That local file is ignored by git.

## Training workflow

Typical search-assist workflow:
- collect live search traces with `ORANGURU_SEARCH_TRACE=1`
- convert JSONL traces with `training/prepare_search_assist_from_jsonl.py`
- build targeted datasets such as switch-only or tera-only subsets
- train helper models with `training/train_search_prior_value.py`
- test any integration behind an env flag with a smoke run first, then repeated 500-battle blocks

There are also older replay-imitation and offline RL scripts in `training/`, but the search engine is the current competitive path.

## Tests

The repo has focused tests under `tests/`, but they are guardrails rather than a polished full-suite CI story. Run the subset relevant to the area you are changing.

Examples:

```bash
source venv/bin/activate
venv/bin/python -m unittest tests/test_rule_bot_heuristics.py
venv/bin/python -m unittest tests/test_engine_safeguards.py
```

## Repo hygiene

Generated artifacts stay local and are ignored by git:
- `logs/`
- `checkpoints/`
- training pickles under `data/`
- local ladder credentials
- ad hoc backup archives under `backups/`

That keeps the tracked repo focused on code, static assets, small reference data, and docs.

## Good entry points

If you are trying to understand the codebase, start here:
1. `src/players/oranguru_engine.py`
2. `src/players/rule_bot.py`
3. `evaluation/eval_vs_foulplay.py`
4. `evaluation/run_block_eval.py`
5. `training/train_search_prior_value.py`

## Reference snapshots

The `backups/` directory contains a few frozen snapshots of historically stronger configurations. They are reference points, not the active development path.
