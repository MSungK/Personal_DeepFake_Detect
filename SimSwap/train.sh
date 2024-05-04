#!/usr/bin/zsh

python train.py \
    --name VGGFaceHQ_512_finetune \
    --batchSize 2 \
    --which_epoch latest \
    --dataset ../data/vggface2-HQ/VGGface2_None_norm_512_true_bygfpgan \
    --gpu_ids 0 \
    --load_pretrain checkpoints/VGGFaceHQ_512_finetune