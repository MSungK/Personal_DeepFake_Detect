#!/usr/bin/zsh
export CUDA_VISIBLE_DEVICES=3
python make_custom_deepfake.py \
    --name sample_data \
    --batchSize 2 \
    --dataset ../data/sample_data \
    --gpu_ids 1 \
    --sample_freq 1 \
    --custom 
