#!/usr/bin/env bash
# GPU job queue runner (docs/CONTRACTS.md C4). Runs *.job files in name order, one at a time.
# Each .job holds ONE shell command line, executed with cwd=perchpilot/.
# State: NN_name.job -> .running -> .done | .failed ; log: NN_name.log
# Exits after IDLE_MAX seconds without new jobs (default 4 h).
Q=/home/sup/data/perchpilot/queue
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IDLE_MAX=${IDLE_MAX:-14400}
mkdir -p "$Q"
idle=0
while [ "$idle" -lt "$IDLE_MAX" ]; do
  job=$(ls "$Q"/*.job 2>/dev/null | sort | head -n 1)
  if [ -z "$job" ]; then
    sleep 20; idle=$((idle + 20)); continue
  fi
  idle=0
  stem="${job%.job}"
  mv "$job" "$stem.running"
  echo "[$(date '+%F %T')] start $(basename "$stem")" >> "$Q/queue.log"
  ( cd "$ROOT" && bash -c "$(cat "$stem.running")" ) > "$stem.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then mv "$stem.running" "$stem.done"; else mv "$stem.running" "$stem.failed"; fi
  echo "[$(date '+%F %T')] end $(basename "$stem") rc=$rc" >> "$Q/queue.log"
done
echo "[$(date '+%F %T')] queue idle, exit" >> "$Q/queue.log"
