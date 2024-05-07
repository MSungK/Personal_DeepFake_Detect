#!/usr/bin/zsh
export CUDA_VISIBLE_DEVICES=3
python train.py \
    --name global_face_224 \
    --batchSize 24 \
    --dataset ../data/global_face \
    --gpu_ids 3 \
    --custom 