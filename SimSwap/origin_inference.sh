#!/usr/bin/zsh

python test_one_image.py \
        --name people \
        --Arc_path arcface_model/arcface_checkpoint.tar \
        --pic_a_path ../data/asian_face/15.png \
        --pic_b_path ../data/asian_face/6581.png \
        --output_path output/ 
        
