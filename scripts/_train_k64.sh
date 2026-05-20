#!/bin/bash
cd /home/leonardo/projects/LLM/byte-language-lab || exit 1
export LD_LIBRARY_PATH="$(pwd)/vendor/llama-cpp/lib:/opt/rocm-7.2.1/lib:$LD_LIBRARY_PATH"
export HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0
pkill -9 -f train_smg_hsa 2>/dev/null
sleep 2
setsid nice -n 10 ./target/release/train_smg_hsa data/corpus_local.txt \
    > data/train_k64_full.log 2>&1 < /dev/null &
echo "launched pid $!"
