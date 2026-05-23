#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/venv/bin/python}"
PLAYER="${PLAYER:-oranguru_engine}"
FORMAT="${FORMAT:-gen9randombattle}"
BATTLES="${BATTLES:-100}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
RUN_LABEL="${RUN_LABEL:-riskguards24_disable_regressed_guards}"
RUN_TAG="${RUN_TAG:-oranguru_static8_800ms_${RUN_LABEL}_${BATTLES}_$(date +%Y%m%d_%H%M%S)}"
RUNNER_LOG="${RUNNER_LOG:-logs/ladder/${RUN_TAG}.runner.log}"

prompt_secret() {
  local var_name="$1"
  local prompt="$2"
  if [ -z "${!var_name:-}" ]; then
    if [ ! -t 0 ]; then
      echo "Missing $var_name and stdin is not interactive; export it or run this script from a terminal." >&2
      exit 2
    fi
    if [ "$var_name" = "PS_PASSWORD" ]; then
      read -r -s -p "$prompt" "$var_name"
      echo
    else
      read -r -p "$prompt" "$var_name"
    fi
    export "$var_name"
  fi
}

prompt_secret PS_USERNAME "Pokemon Showdown username: "
prompt_secret PS_PASSWORD "Pokemon Showdown password: "
PS_USERNAME="$(printf '%s' "$PS_USERNAME" | tr '[:upper:]' '[:lower:]')"
export PS_USERNAME

mkdir -p logs/ladder logs/search_traces/current logs/ladder_review training_data/ladder_loss_pretrain
printf '%s\n' "$RUNNER_LOG" > logs/ladder/latest_headless_log.txt

ARGS=(
  --player "$PLAYER"
  --format "$FORMAT"
  --battles "$BATTLES"
  --max-concurrent 1
  --progress-every "$PROGRESS_EVERY"
  --auto-reconnect
  --rejoin-active
  --bot-version "$RUN_TAG"
  --ladder-log "logs/ladder/${RUN_TAG}.jsonl"
  --snapshot-log "logs/ladder/${RUN_TAG}.snapshots.jsonl"
)

export ORANGURU_BOT_VERSION="${ORANGURU_BOT_VERSION:-$RUN_TAG}"
export ORANGURU_SEARCH_MS="${ORANGURU_SEARCH_MS:-800}"
export ORANGURU_SAMPLE_STATES="${ORANGURU_SAMPLE_STATES:-8}"
export ORANGURU_SAMPLE_STATES_MAX="${ORANGURU_SAMPLE_STATES_MAX:-8}"
export ORANGURU_DYNAMIC_SAMPLING="${ORANGURU_DYNAMIC_SAMPLING:-0}"
export ORANGURU_MCTS_DETERMINISTIC="${ORANGURU_MCTS_DETERMINISTIC:-0}"
export ORANGURU_LOW_HP_DEFENSIVE_TOP_GUARD="${ORANGURU_LOW_HP_DEFENSIVE_TOP_GUARD:-0}"
export ORANGURU_NONBENEFICIAL_ATTACK_TERA_GUARD="${ORANGURU_NONBENEFICIAL_ATTACK_TERA_GUARD:-0}"
export ORANGURU_RAPID_SPIN_VALUE_GUARD="${ORANGURU_RAPID_SPIN_VALUE_GUARD:-0}"
export ORANGURU_PIVOT_CHURN_GUARD="${ORANGURU_PIVOT_CHURN_GUARD:-0}"
export ORANGURU_FATAL_REPLY_GUARD="${ORANGURU_FATAL_REPLY_GUARD:-1}"
export ORANGURU_CONTACT_RISK_GUARD="${ORANGURU_CONTACT_RISK_GUARD:-1}"
export ORANGURU_TERA_DEFENSIVE_SANITY="${ORANGURU_TERA_DEFENSIVE_SANITY:-1}"
export ORANGURU_ANTI_SWEEPER_CONTROL_GUARD="${ORANGURU_ANTI_SWEEPER_CONTROL_GUARD:-1}"
export ORANGURU_OPP_BEHAVIOR_MODEL="${ORANGURU_OPP_BEHAVIOR_MODEL:-1}"
export ORANGURU_OUTSPEED_KO_SWITCH_GUARD="${ORANGURU_OUTSPEED_KO_SWITCH_GUARD:-1}"
export ORANGURU_OUTSPEED_KO_SWITCH_FACTOR="${ORANGURU_OUTSPEED_KO_SWITCH_FACTOR:-1.15}"
export ORANGURU_SEARCH_TRACE="${ORANGURU_SEARCH_TRACE:-1}"
export ORANGURU_SEARCH_TRACE_OUT="${ORANGURU_SEARCH_TRACE_OUT:-logs/search_traces/current/${RUN_TAG}.jsonl}"
export ORANGURU_SEARCH_TRACE_TAG="${ORANGURU_SEARCH_TRACE_TAG:-$RUN_TAG}"
export ORANGURU_SEARCH_TRACE_INCLUDE_FP_ORACLE="${ORANGURU_SEARCH_TRACE_INCLUDE_FP_ORACLE:-1}"

nohup "$PYTHON_BIN" evaluation/ladder_rulebot.py "${ARGS[@]}" > "$RUNNER_LOG" 2>&1 &
PID="$!"
printf '%s\n' "$PID" > logs/ladder/headless.pid

echo "$RUN_TAG"
echo "pid=$PID log=$RUNNER_LOG"
tail -f "$RUNNER_LOG"
