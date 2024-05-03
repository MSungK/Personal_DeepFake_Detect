#!/usr/bin/zsh

python train.py --name simswap512_test  --batchSize 16  --gpu_ids 0 --dataset /path/to/VGGFace2HQ --Gdeep True
