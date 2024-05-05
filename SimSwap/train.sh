#!/usr/bin/zsh

python train.py \
    --name k-face \
    --batchSize 2 \
    --dataset ../data/k-face \
    --gpu_ids 0 \
    --custom 