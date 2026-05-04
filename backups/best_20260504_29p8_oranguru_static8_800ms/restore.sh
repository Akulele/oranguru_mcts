#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SRC_DIR}/../.." && pwd)"

tar -xzf "${SRC_DIR}/source_cbe81ac.tar.gz" -C "${ROOT_DIR}"

echo "Restored tracked source snapshot from ${SRC_DIR}/source_cbe81ac.tar.gz"
