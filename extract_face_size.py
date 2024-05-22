from ultralytics import YOLO
from PIL import Image
import os
from os import path as osp
import torch
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    model = 'weight/face_detector/weights/best.pt'
    model = YOLO(model)
    src_path = 'data/asian_face'
    mean_list = list()
    min_box_img_path = ''
    min_box_size = 10000000
    
    f = open('not_detected.txt', 'w')

    for img_path in tqdm(os.listdir(src_path)):
        img_path = osp.join(src_path, img_path)
        img = Image.open(img_path)
        results = model.predict(source=[img], device='cuda:0', iou=0.01)
        try:
            w, h = torch.tensor(results[0].boxes.xywh).cpu().squeeze(0)[2:]
            w = int(w) 
            h = int(h)
            mean_list.append([w,h])
            if w*h < min_box_size:
                min_box_size = w*h
                min_box_img_path = results[0].path
        except:
            f.write(results[0].path + '\n')

    mean_list = np.array(mean_list)
    np.save('box_shape', mean_list)
    print(min_box_img_path)
    

        # print(results[0].path)


# Tracking for Video
    # results = model.track(source="https://youtu.be/LNwODJXcvt4", device='cuda:0', show=True, save_crop=True)
    # results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")
