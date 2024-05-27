import torch
import numpy as np
from os.path import join
from dataset import AbstractDataset
from PIL import Image
from torchvision import transforms as T
from logging import info
import os
from os import path as osp
from torch.utils.data import Dataset

class Custom_Dataset(Dataset):
    def __init__(self, root_path):
        super(Custom_Dataset, self).__init__()
        self.real_images = list()
        self.fake_images = list()
        for file in os.listdir(osp.join(root_path, 'real')):
            file = osp.join(root_path, 'real', file)
            self.real_images.append([file, 1])
        for file in os.listdir(osp.join(root_path, 'fake')):
            file = osp.join(root_path, 'fake', file)
            self.fake_images.append([file, 0])
        self.images = self.real_images + self.fake_images

        info(f"Stage : {root_path.split('/')[-1]}")
        if root_path.split('/')[-1] == 'train':
            self.transform = T.Compose([T.Resize(299),
                                        T.RandomHorizontalFlip(p=0.5),
                                        T.ToTensor()])
        else:
            self.transform = T.Compose([T.Resize(299),
                                        T.ToTensor()])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img, label = self.images[index]
        img = self.transform(Image.open(img))
        label = torch.tensor(label) # TODO
        return img, label