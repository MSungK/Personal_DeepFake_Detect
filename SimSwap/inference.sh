#!/usr/bin/zsh

python make_deepfake.py \
        --name people \
        --Arc_path arcface_model/arcface_checkpoint.tar \
        --source_path ../data/global_face/860.png \
        --target_path ../data/global_face/23.png \
        --crop_size 224 \
        --output_path generated/