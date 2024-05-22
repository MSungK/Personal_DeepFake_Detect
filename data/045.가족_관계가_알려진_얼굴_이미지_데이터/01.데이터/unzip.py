import zipfile
import os
import os.path as osp


if __name__ == '__main__':
    path = '1.Training/원천데이터_0903_add'
    assert osp.exists(path), f'{path} does not exist'
    for zip_file in os.listdir(path):
        zip_file = osp.join(path, zip_file)
        if osp.isdir(zip_file): continue
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(path)
            os.remove(zip_file)
        print(zip_file)