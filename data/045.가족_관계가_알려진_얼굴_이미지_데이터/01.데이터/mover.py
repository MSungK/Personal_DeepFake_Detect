import os
import os.path as osp
from glob import glob
from shutil import move
import logging
from tqdm import tqdm


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()])


if __name__ == '__main__':
    setup_logger()
    path = '1.Training'
    img_path = '1.Training/원천데이터'
    label_path = '1.Training/라벨링데이터'
    for file in tqdm(glob(f'{path}/**/*', recursive=True)):
        if osp.isdir(file): continue
        file_name = file.split('/')[-1]
        if osp.splitext(file_name)[-1] in ['.JSON', '.json']:
            move(file, osp.join(label_path, file_name))
        elif osp.splitext(file_name)[-1] in ['.png', '.PNG', '.jpg', '.JPG']:
            move(file, osp.join(img_path, file_name))
        else:
            logging.info(file)