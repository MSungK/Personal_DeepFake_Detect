import os
import os.path as osp
import json
from tqdm import tqdm
from shutil import move

if __name__ == '__main__':
    
    path_list = ['1.Training/라벨링데이터', '2.Validation/라벨링데이터/A(─ú░í)/2.Individuals', '2.Validation/라벨링데이터/B(┐▄░í)/2.Individuals']
    tgt_path = 'preprocessed'
    os.makedirs(tgt_path)
    filename_format = 'aihub_iamge'
    filename_counter = 10000000
    
    for path in path_list:
        for file in tqdm(os.listdir(path)):
            json_path = osp.join(path, file)
            with open(json_path) as f:
                json_file = json.load(f)
            if len(json_file['member']) == 1 and json_file['member'][0]['angle'] == '0':
                img_path = osp.join(path.replace('라벨링데이터', '원천데이터'), json_file['filename'])
                if not osp.exists(img_path):
                    print(f'{img_path} does not exist!!!')
                    continue
                move(img_path, osp.join(tgt_path, filename_format+str(filename_counter)+osp.splitext(json_file['filename'])[1]))
                filename_counter+=1
    
    print(filename_counter)