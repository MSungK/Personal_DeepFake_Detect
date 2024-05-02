import os
import os.path as osp
from glob import glob
from tqdm import tqdm

'''
['0 0.1953125 0.4270833333333333 0.109375 0.19270833333333334\n', '0 0.8408203125 0.5377604166666666 0.095703125 0.21614583333333334\n']
'''

if __name__ == '__main__':
    root_dir = 'wider_yolo_format/train/labels'
    gt_paths = glob(osp.join(root_dir, '*.txt'))
    print(len(gt_paths))
    for gt_path in tqdm(gt_paths):
        f = open(gt_path, 'r')
        lines = f.readlines()
        new_lines = list()
        for line in lines:
            line = list(map(float, line.strip().split(' ')))
            # print(line)
            line[3] *= 1.4
            line[4] *= 1.4
            line[0] = int(line[0])
            # print(line)
            assert len(line) == 5
            line = list(map(str, line))
            new_line = ' '.join(line)
            new_line += '\n'
            # print(new_line)
            new_lines.append(new_line)
        f.close()
        new_tmp = ''.join(new_lines)
        f = open(gt_path, 'w')
        f.write(new_tmp)
        f.close()

    