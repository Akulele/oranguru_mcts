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
- broad neural action steering is not trusted yet
- targeted helpers are more promising than general learned move selection

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
