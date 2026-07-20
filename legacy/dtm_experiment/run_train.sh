#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/train_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training in the background."
echo "  log: $LOG"
echo "  Note: keep the lid open (or plugged in + external display) -- see"
echo "        the comment at the top of this script for why."

nohup caffeinate -dimsu python3 "$HERE/train.py" > "$LOG" 2>&1 &
PID=$!
disown

echo "  pid: $PID"
echo ""
echo "Tail progress with:"
echo "  tail -f \"$LOG\""
echo "Stop it with:"
echo "  kill $PID"
