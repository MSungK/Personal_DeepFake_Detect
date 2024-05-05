import os
import os.path as osp
from shutil import move
from collections import deque

if __name__ == '__main__':
    tgt_path = 'k-face'
    src_path = 'k-face'
    files = list()
    q = deque()
    q.appendleft(src_path)
    cnt = 0
    while len(q)!=0:
        cur_path = q.pop()
        for next_path in os.listdir(cur_path):
            next_path = osp.join(cur_path, next_path)
            if osp.isfile(next_path) and osp.splitext(next_path)[1] in ['.jpg', '.png']:
                cnt += 1
            elif osp.isdir(next_path):
                print(next_path)
                os.system(f'rm -r {next_path}')
            # else:
            #     print(f"There is a wrong file {next_path}")
    print("TOTAL : {}".format(cnt))