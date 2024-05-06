#!/usr/bin/zsh
export CUDA_VISIBLE_DEVICES=3
python train.py \
    --name k-face_224 \
    --batchSize 24 \
    --dataset ../data/k-face \
    --gpu_ids 3 \
    --custom
    --sample_freq 5000
    --model_freq 5000