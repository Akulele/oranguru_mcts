#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SRC_DIR}/../.." && pwd)"

cp -a "${SRC_DIR}/oranguru_engine.py" "${ROOT_DIR}/src/players/oranguru_engine.py"
cp -a "${SRC_DIR}/rule_bot.py" "${ROOT_DIR}/src/players/rule_bot.py"
cp -a "${SRC_DIR}/eval_vs_foulplay.py" "${ROOT_DIR}/evaluation/eval_vs_foulplay.py"

if [[ -f "${SRC_DIR}/logs/foulplay/eval_20260123_183550_base.log" ]]; then
  mkdir -p "${ROOT_DIR}/logs/foulplay"
  cp -a "${SRC_DIR}/logs/foulplay/eval_20260123_183550_base.log" "${ROOT_DIR}/logs/foulplay/"
fi

echo "Restored snapshot from ${SRC_DIR} to ${ROOT_DIR}"
