# Best Oranguru Snapshot - 2026-05-04

Snapshot of the most performant confirmed Oranguru version before shifting to cost-reduction work.

## Commit

`cbe81ac` - Trace critical recovery reranks

The worktree was cleaned before backup. Pre-existing dirty/untracked changes were preserved in git stash:

`pre-backup dirty worktree 20260504_142121`

## Best Known Eval

Configuration family: pure MCTS, finish-blow guard only, static 8 worlds, 800ms per world.

Key settings:

```text
ORANGURU_SEARCH_MS=800
ORANGURU_SAMPLE_STATES=8
ORANGURU_SAMPLE_STATES_MAX=8
ORANGURU_DYNAMIC_SAMPLING=0
ORANGURU_FINISH_BLOW_GUARD=1
ORANGURU_CRITICAL_RECOVERY_GUARD=0
ORANGURU_SETUP_WINDOW=0
ORANGURU_RECOVERY_WINDOW=0
ORANGURU_PROGRESS_WINDOW=0
ORANGURU_SWITCH_GUARD=0
ORANGURU_TACTICAL_SHADOW_WINDOWS=0
ORANGURU_RL_PRIOR=0
ORANGURU_SEARCH_PRIOR=0
ORANGURU_SWITCH_PRIOR=0
ORANGURU_TERA_PRUNER=0
ORANGURU_LEAF_VALUE=0
ORANGURU_WORLD_RANKER=0
ORANGURU_RERANK_GATE=0
```

Confirmed results from user-run blocks:

```text
finish_blow_pure_mcts_800ms_8states_static_4x250: 295/1000 = 29.5%
finish_blow_pure_mcts_800ms_8states_static_confirm_4x250 valid blocks only: 227/750 = 30.27%
Combined valid evidence: 522/1750 = 29.83%
```

## Contents

`source_cbe81ac.tar.gz` is a git archive of committed `HEAD` at backup time.

Use `restore.sh` from repo root if you need to restore this tracked source snapshot.
