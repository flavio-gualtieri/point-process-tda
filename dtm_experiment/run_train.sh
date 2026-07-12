#!/usr/bin/env bash
# Runs dtm_experiment/train.py detached from the terminal, so it survives
# closing the terminal window/tab or an SSH session dropping, and wrapped in
# `caffeinate` so macOS won't idle-sleep, display-sleep, or disk-sleep the
# Mac while it's running.
#
# CAVEAT: caffeinate cannot override clamshell (lid-closed) sleep. If you
# physically close the lid without an external display attached, macOS will
# still suspend the whole machine at the hardware level and this process
# will pause too. To survive a closed lid, keep the Mac plugged into power
# with an external display connected; otherwise just leave the lid open
# (the display staying on is expected -- caffeinate -d prevents display
# sleep on purpose, so idle-sleep/App Nap don't throttle the training loop).
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
