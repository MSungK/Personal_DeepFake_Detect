import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob as g
import cv2
from tqdm import tqdm
from shutil import copy, move
import os.path as osp

if __name__ == '__main__':

	new_imgs_dir = 'wider_yolo_format/images/valid'
	new_lbls_dir = 'wider_yolo_format/labels/valid'
	label_text_name = 'wider_face/wider_face_split/wider_face_val_bbx_gt.txt'
	imgs_address = 'wider_face/WIDER_val/images'
	scale=1.5

	os.makedirs(new_imgs_dir,exist_ok = True)
	os.makedirs(new_lbls_dir,exist_ok = True)
	annots = open(label_text_name) 
	lines = annots.readlines()
	names =   [x for x in lines if 'jpg' in x]
	indices = [lines.index(x) for x in names]


	for n in tqdm(range(len(names[:]))):
		i = indices[n]
		name = lines[i].rstrip()
		old_img_path = os.path.join(imgs_address , name)
		name = name.split('/')[-1]
		label_path = os.path.join(new_lbls_dir , name.split('.')[0] + '.txt')
		img_path = os.path.join(new_imgs_dir , name)
		
		num_objs = int(lines[i+1].rstrip())		
		bboxs = lines[i+2 : i+2+num_objs]
		bboxs = list(map(lambda x:x.rstrip() , bboxs))
		bboxs = list(map(lambda x:x.split()[:4], bboxs))
		# if len(bboxs) > 5:
		#     continue
		img = cv2.imread(old_img_path)
		img_h,img_w,_ = img.shape
		img_h,img_w,_ = img.shape
		f = open(label_path, 'w')
		count = 0 # Num of bounding box
		for bbx in bboxs:
			flag = True
			x1 = int(bbx[0])
			y1 = int(bbx[1])
			w = int(bbx[2])
			h = int(bbx[3])
			if int(x1+(w*scale)) >= img_w:
				flag = False
			if int(y1+(h*scale)) >= img_h:
				flag = False
		#     #yolo:
			x = (x1 + w//2) / img_w
			y = (y1 + h//2) / img_h
			if flag:
				w = (w*scale) / img_w
				h = (h*scale) / img_h
			else:
				w = w / img_w
				h = h / img_h
			if w * h * 100 > 2:
				yolo_line = f'{0} {x} {y} {w} {h}\n'
				f.write(yolo_line)
				count += 1
		f.close()
		if count > 0:   
			copy(old_img_path , img_path)
		else:
			os.remove(label_path)