#!/bin/bash
# Launch ONE Qwen3 engine on :3000, detached. Single engine — 2 engines on
# one RDNA2 GPU crashed under load (HIP 700).
LAB="/home/leonardo/projects/LLM/llm-lab"
cd "$LAB" || exit 1
export LD_LIBRARY_PATH="$LAB/vendor/llama-cpp/lib:/opt/rocm-7.2.1/lib"
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0
pkill -9 -f iara-engine 2>/dev/null
sleep 5
IARA_HTTP_PORT=3000 setsid nice -n 10 ./target/release/iara-engine "$LAB/models" \
    > "$LAB/eng3000.log" 2>&1 < /dev/null &
echo "engine :3000 pid $!"
