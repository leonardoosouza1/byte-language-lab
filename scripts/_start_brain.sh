#!/bin/bash
# Launch the brain server detached, with a working log path.
cd /home/leonardo/projects/LLM/byte-language-lab || exit 1
pkill -9 -f iara_brain_full 2>/dev/null
sleep 3
LOG="/home/leonardo/projects/LLM/byte-language-lab/data/brain_serve.log"
setsid nice -n 10 ./target/release/iara_brain_full serve > "$LOG" 2>&1 < /dev/null &
echo "launched pid $! → $LOG"
