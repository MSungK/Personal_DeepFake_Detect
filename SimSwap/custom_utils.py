import logging
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import glob
import os.path as osp
from PIL import Image
import random
import torch

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()],
    )


class data_prefetcher():
    def __init__(self, loader):
        self.loader = loader
        self.dataiter = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.num_images = len(loader)
        self.preload()

    def preload(self):
        try:
            self.src_image1, self.src_image2 = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            self.src_image1, self.src_image2 = next(self.dataiter)
            
        with torch.cuda.stream(self.stream):
            self.src_image1  = self.src_image1.cuda(non_blocking=True)
            self.src_image1  = self.src_image1.sub_(self.mean).div_(self.std)
            self.src_image2  = self.src_image2.cuda(non_blocking=True)
            self.src_image2  = self.src_image2.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        src_image1  = self.src_image1
        src_image2  = self.src_image2
        self.preload()
        return src_image1, src_image2
    
    def __len__(self):
        """Return the number of images"""
        return self.num_images


class K_Dataset(Dataset):
    def __init__(self, root_path, transform, random_seed):
        self.image_dir=root_path
        self.img_transform=transform
        self.seed=random_seed
        self.files=list()
        for file in glob.glob(f'{root_path}/*'):
            if osp.splitext(file)[1] in ['.jpg', '.png']:
                self.files.append(file)
        self.len_dataset = len(self.files)
        random.seed(self.seed)

    def __getitem__(self, index):
        src_image = self.img_transform(Image.open(self.files[index]))
        tgt_image = self.img_transform(Image.open(self.files[random.randint(0, self.len_dataset-1)]))
        return src_image, tgt_image
    
    def __len__(self):
        return self.len_dataset
    
    
def K_DataLoader(   root_path, 
                    batch_size=16,
                    num_workers=4,
                    random_seed=1234
                ):
    c_transforms=T.Compose([
        T.Resize(224),
        T.ToTensor()
    ])
    dataset = K_Dataset(root_path, c_transforms, random_seed)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            drop_last=True, shuffle=True, num_workers=num_workers, pin_memory=True)
    prefetcher = data_prefetcher(dataloader)
    return prefetcher