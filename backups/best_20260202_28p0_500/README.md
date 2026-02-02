# Best snapshot (2026-02-02, 28.0% win rate vs Foul Play)

Eval summary (server):
- Battles: 500
- W/L/T: 140/360/0
- Win rate: 28.0%
- Avg turns: 28.8
- Foul Play log: /mnt/ai/AI_Lab/oranguru_mcts/logs/foulplay/current/eval_20260202_061219_detfix500.log

MCTS diagnostics:
- Calls/states: 14378/191064
- Empty results: 0 (0.0%)
- State fails: 0 (0.0%)
- Deterministic/Stochastic: 14378/0
- Fallback(super/random): 0/13 (0.0%/0.1%)

Command used (server):
```
FOULPLAY_PY="$(pwd)/third_party/foul-play/.venv/bin/python"
RUN_TS="$(date +%Y%m%d_%H%M%S)_detfix500"
ROOT="$(pwd)"

ORANGURU_SHOWDOWN_WS="ws://127.0.0.1:8000/showdown/websocket" \
ORANGURU_BELIEF_SAMPLING=1 \
ORANGURU_MCTS_DETERMINISTIC=1 ORANGURU_MCTS_DETERMINISTIC_EVAL_ONLY=1 \
ORANGURU_UNLIKELY_CHOICE_INFER=1 \
ORANGURU_PARALLELISM=6 ORANGURU_SAMPLE_STATES=6 ORANGURU_SEARCH_MS=350 \
ORANGURU_AUTO_TERA=0 ORANGURU_HEURISTIC_BLEND=0.0 ORANGURU_MIN_HEURISTIC_BLEND=0.0 \
ORANGURU_SELECTION_MODE=blend ORANGURU_GATE_MODE=hard ORANGURU_MCTS_CONFIDENCE=0.3 \
ORANGURU_STATUS_KO_GUARD=0 \
venv/bin/python evaluation/eval_vs_foulplay.py \
  --player oranguru_engine --battles 500 --format gen9randombattle \
  --foulplay-python "$FOULPLAY_PY" \
  --foulplay-username foulplay_bot_$RUN_TS --foulplay-password foulplay_pass \
  --foulplay-search-ms 120 --foulplay-parallelism 1 --foulplay-wait 90 \
  --battle-timeout 240 --foulplay-retries 5 --no-login \
  --foulplay-log "$ROOT/logs/foulplay/current/eval_$RUN_TS.log" \
  --foulplay-user-id-file "$ROOT/logs/foulplay/current/user_$RUN_TS.txt" \
  --ws-uri ws://127.0.0.1:8000/showdown/websocket | tee "$ROOT/logs/evals/current/$RUN_TS.stdout.log"
```

Preferred parser command:
```
python evaluation/aggregate_mcts_diag.py --glob "logs/evals/current/*.stdout.log"
```

Files captured:
- src/players/oranguru_engine.py
- src/players/rule_bot.py
- evaluation/eval_vs_foulplay.py
