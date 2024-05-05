#!/usr/bin/zsh

python test_one_image.py \
        --name people \
        --Arc_path arcface_model/arcface_checkpoint.tar \
        --pic_a_path crop_224/6.jpg \
        --pic_b_path crop_224/ds.jpg \
        --crop_size 512 \
        --output_path output/ 