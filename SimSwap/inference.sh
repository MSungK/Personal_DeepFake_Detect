#!/usr/bin/zsh

python test_one_image.py \
        --name people \
        --Arc_path arcface_model/arcface_checkpoint.tar \
        --pic_a_path ../data/global_face/860.png \
        --pic_b_path ../data/global_face/23.png \
        --crop_size 224 \
        --output_path custom_output/ 
        