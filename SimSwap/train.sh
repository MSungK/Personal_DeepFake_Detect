#!/usr/bin/zsh

# python train.py --name simswap512_test  \
#                 --batchSize 16 \
#                 --gpu_ids 0 \
#                 --dataset /path/to/VGGFace2HQ \
#                 --Gdeep True \
#                 --load_pretrain ./checkpoints

python train.py \
    --name simswap224_test \
    --batchSize 8  \
    --gpu_ids 0 \
    --dataset ../data/vggface_224 \
    --Gdeep False \
    --log_frep 100 \
    --continue_train False \
    --sample_freq 100 \
    --custom \
    --G_path checkpoints/people/latest_net_G.pth