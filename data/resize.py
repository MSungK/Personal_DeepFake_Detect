from PIL import Image
import os.path as osp
from glob import glob
from tqdm import tqdm
from shutil import move

if __name__ == '__main__':
    dir_path = 'global_face/'
    tgt_path = 'global_face/'
    tmp = set()
    for file_path in tqdm(glob(dir_path + '/*.jpg')):
        # tgt_name = file_path.split('/')[-1]
        # print(file_path)
        # exit()
        # move(file_path, osp.join(tgt_path, tgt_name))
        img = Image.open(file_path)
        assert img.size == (224, 224), f'img.size : {img.size}'
        tmp.add(osp.splitext(file_path)[-1].split('.')[-1])
    for file_path in tqdm(glob(dir_path + '/*.png')):
        # tgt_name = file_path.split('/')[-1]
        # print(file_path)
        # exit()
        # move(file_path, osp.join(tgt_path, tgt_name))
        img = Image.open(file_path)
        try:
            assert img.size == (224, 224), f'img.size : {img.size}'
        except:
            img = img.resize((224,224))
            img.save(file_path)
            tmp.add(osp.splitext(file_path)[-1].split('.')[-1])
    print(tmp)
