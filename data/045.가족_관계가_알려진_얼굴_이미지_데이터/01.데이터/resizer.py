
import os
import os.path as osp
from tqdm import tqdm
import cv2


if __name__ == '__main__':
    src_path = 'preprocessed'    
    tgt_path = 'resized'
    for filename in tqdm(os.listdir(src_path)):
        filepath = osp.join(src_path, filename)
        img = cv2.imread(filepath)
        img = cv2.resize(img, (224,224))
        cv2.imwrite(osp.join(tgt_path, filename), img)