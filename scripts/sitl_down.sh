#!/usr/bin/env bash
# Stop ONLY the processes started by sitl_up.sh (recorded PIDs), never anything else.
RUN=/home/sup/data/perchpilot/sitl
[ -f "$RUN/pids" ] || { echo "no pid file"; exit 0; }
killed=""
while read -r name pid; do
  if kill -0 "$pid" 2>/dev/null; then
    # double-check it is ours: gz must carry our partition, px4 must be instance 7
    env_ok=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -c "GZ_PARTITION=perchpilot" || true)
    if [ "$env_ok" -ge 1 ]; then kill "$pid" && echo "stopped $name $pid" && killed="$killed $pid"; else echo "skip $name $pid (not ours)"; fi
  fi
done < "$RUN/pids"
# gz sim forks a server child with our partition; stop those too
for p in $(pgrep -f "^gz sim -r -s --headless-rendering $RUN/perch_world\.sdf( |$)" || true); do
  tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "GZ_PARTITION=perchpilot" && kill "$p" && echo "stopped gz child $p" \
    && killed="$killed $p"
done
# review D14: wait (max 15 s) until the stopped processes are really gone before the pid file is cleared
for _ in $(seq 1 30); do
  alive=0
  for p in $killed; do kill -0 "$p" 2>/dev/null && alive=1; done
  [ "$alive" = 0 ] && break
  sleep 0.5
done
: > "$RUN/pids"
