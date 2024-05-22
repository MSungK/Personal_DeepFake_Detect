from ultralytics import YOLO
from PIL import Image
import os
from os import path as osp
import torch
import numpy as np
from tqdm import tqdm
from shutil import move

if __name__ == '__main__':
    model = 'weight/face_detector/weights/best.pt'
    model = YOLO(model)
    src_path = 'data/asian_face'
    tgt_path = 'data/filtered'
    ref_size = 80 * 140

    f = open('not_detected.txt', 'w')

    for img_name in tqdm(os.listdir(src_path)):
        img_path = osp.join(src_path, img_name)
        img = Image.open(img_path)
        results = model.predict(source=[img], device='cuda:0', iou=0.01)
        try:
            w, h = torch.tensor(results[0].boxes.xywh).cpu().squeeze(0)[2:]
            w = int(w) 
            h = int(h)
            if w*h < ref_size:
                move(img_path, osp.join(tgt_path, img_name))
        except:
            f.write(results[0].path + '\n')

        # print(results[0].path)


# Tracking for Video
    # results = model.track(source="https://youtu.be/LNwODJXcvt4", device='cuda:0', show=True, save_crop=True)
    # results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")
