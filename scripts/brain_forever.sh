#!/bin/bash
# Always-on supervisor: restarts brain if crashes. Run in screen/tmux.
# Usage: ./brain_forever.sh [serve|...]
# Stop: kill the wrapper PID OR touch /tmp/brain_stop_flag

set -u
BRAIN_DIR="/home/leonardo/projects/LLM/byte-language-lab"
BIN="$BRAIN_DIR/target/release/iara_brain_full"
ARG="${1:-serve}"
LOG_DIR="/tmp/brain_logs"
STOP_FLAG="/tmp/brain_stop_flag"
mkdir -p "$LOG_DIR"
rm -f "$STOP_FLAG"

echo "▸ Brain forever supervisor started"
echo "  bin: $BIN"
echo "  log: $LOG_DIR"
echo "  stop: touch $STOP_FLAG"
echo

cd "$BRAIN_DIR"

restart_count=0
while true; do
    if [ -f "$STOP_FLAG" ]; then
        echo "▸ Stop flag detected. Exiting supervisor."
        rm -f "$STOP_FLAG"
        break
    fi
    ts=$(date +%s)
    log="$LOG_DIR/brain_$ts.log"
    echo "$(date -Iseconds) ▸ Starting brain (restart #$restart_count) → $log"
    "$BIN" "$ARG" >> "$log" 2>&1
    exit_code=$?
    echo "$(date -Iseconds) ✗ Brain exited code=$exit_code, restarting in 3s..."
    restart_count=$((restart_count + 1))
    sleep 3
done

echo "▸ Supervisor done. Total restarts: $restart_count"
