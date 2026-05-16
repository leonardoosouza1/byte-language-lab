#!/bin/bash
# Launch both Qwen3 engines detached with working log paths.
LAB="/home/leonardo/projects/LLM/llm-lab"
cd "$LAB" || exit 1
export LD_LIBRARY_PATH="$LAB/vendor/llama-cpp/lib:/opt/rocm-7.2.1/lib"
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0

IARA_HTTP_PORT=3000 setsid nice -n 10 ./target/release/iara-engine "$LAB/models" \
    > "$LAB/eng3000.log" 2>&1 < /dev/null &
echo "engine :3000 pid $!"
sleep 3
IARA_HTTP_PORT=3002 IARA_WS_DISABLE=1 setsid nice -n 10 ./target/release/iara-engine "$LAB/models" \
    > "$LAB/eng3002.log" 2>&1 < /dev/null &
echo "engine :3002 pid $!"
