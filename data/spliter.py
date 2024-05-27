import os.path as osp
import os
from glob import glob
from random import shuffle
from shutil import move


if __name__ == '__main__':
    real_path = 'deepfake_classification/asian_face'
    fake_path = 'deepfake_classification/fake_images'
    real_path = glob(real_path + '/*')   
    fake_path = glob(fake_path + '/*')

    print(f'total real images : {len(real_path)}')
    print(f'total fake images : {len(fake_path)}')
    
    shuffle(real_path)
    shuffle(fake_path)
    
    train_path = 'deepfake_classification/train'
    val_path = 'deepfake_classification/val'
    test_path = 'deepfake_classification/test'
    
    train_real = 0
    train_fake = 0
    val_real = 0
    val_fake = 0
    test_real = 0
    test_fake = 0
    
    for i in real_path[:int(len(real_path)*0.7)]:
        move(i, osp.join(train_path,'real',i.split('/')[-1]))
        train_real += 1
    for i in fake_path[:int(len(fake_path)*0.7)]:
        move(i, osp.join(train_path,'fake',i.split('/')[-1]))
        train_fake += 1
    
    for i in real_path[int(len(real_path)*0.7):int(len(real_path)*0.9)]:
        move(i, osp.join(val_path,'real',i.split('/')[-1]))
        val_real += 1
    for i in fake_path[int(len(fake_path)*0.7):int(len(fake_path)*0.9)]:
        move(i, osp.join(val_path,'fake',i.split('/')[-1]))
        val_fake += 1
        
    for i in real_path[int(len(real_path)*0.9):]:
        move(i, osp.join(test_path,'real',i.split('/')[-1]))
        test_real += 1
    for i in fake_path[int(len(fake_path)*0.9):]:
        move(i, osp.join(test_path,'fake',i.split('/')[-1]))
        test_fake += 1
        
    print(f'train_real : {train_real}')
    print(f'train_fake : {train_fake}')
    print(f'val_real : {val_real}')
    print(f'val_fake : {val_fake}')
    print(f'test_real : {test_real}')
    print(f'test_fake : {test_fake}')
    print(f'total : {train_real + train_fake + val_real + val_fake + test_real + test_fake}')