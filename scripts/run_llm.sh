#!/bin/bash
master_port=00097
num_process=1
models=("GPT2")
# 定义 horizon 列表
horizons=(10 20 30 40 50 60)

for model in "${models[@]}"; do
    for h in "${horizons[@]}"; do
        input_size=$((h * 2))
        echo "Running model $model with horizon $h and input_size $input_size"
        accelerate launch  --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_llm.py \
            --model_name "$model" \
            --horizon "$h" \
            --input_size "$input_size"
    done
done