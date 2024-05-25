#!/usr/bin/zsh
export CUDA_VISIBLE_DEVICES=3
python make_custom_deepfake.py \
    --name sample_data \
    --batchSize 24 \
    --dataset ../data/asian_face \
    --gpu_ids 3 \
    --model_path checkpoints/345000_net_G.pth \
    --custom 