#!/usr/bin/zsh

python test_one_image.py \
        --name people \
        --Arc_path arcface_model/arcface_checkpoint.tar \
        --pic_a_path ../data/asian_face/AM123.jpg \
        --pic_b_path ../data/asian_face/AM1525.jpg \
        --output_path output/ 
        
