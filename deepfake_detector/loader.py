from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL import Image
import torchvision.transforms as T
import torch

class Real_Dataset(Dataset):
    def __init__(self, root_path):
        '''
        root_path:
            real
            fake
        '''
        self.reals = glob(f'{root_path}/real/*.png') + glob(f'{root_path}/real/*jpg')
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        self.transform = T.Compose([
            T.Resize(512,512),
            T.Normalize(mean=mean, std=std),
            T.ToTensor()
        ])
    
    def __getitem__(self, index):
        img = self.reals[index]
        img = self.transform(Image.open(img))
        label = torch.tensor(1)
        return img, label

    def __len__(self):
        return len(self.reals)
    
        
class Fake_Dataset(Dataset):
    def __init__(self, root_path):
        '''
        root_path:
            real
            fake
        '''
        self.fakes = glob(f'{root_path}/fake/*.png') + glob(f'{root_path}/fake/*jpg')
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        self.transform = T.Compose([
            T.Resize(512,512),
            T.Normalize(mean=mean, std=std),
            T.ToTensor()
        ])
    
    def __getitem__(self, index):
        img = self.fakes[index]
        img = self.transform(Image.open(img))
        label = torch.tensor(0)
        return img, label

    def __len__(self):
        return len(self.fakes)
    
    
def Loader(dataset, num_workers, is_train:bool):
    shuffle = True if is_train=True else False
    loader = DataLoader(dataset, num_workers=num_workers, pin_memory=True, shuffle=shuffle)
    return loader