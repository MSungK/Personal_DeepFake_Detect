#!/usr/bin/zsh
export CUDA_VISIBLE_DEVICES=3
python train.py \
    --name asian_face \
    --batchSize 24 \
    --dataset ../data/asian_face \
    --gpu_ids 3 \
    --lr 0.00004 \
    --sample_freq 5000 \
    --model_freq 5000 \
    --custom 
