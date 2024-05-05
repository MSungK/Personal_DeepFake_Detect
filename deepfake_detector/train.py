import os
import torch
import torchvision
from utils import setup_logger, parser
import loader
import matplotlib.pyplot as plt


if __name__ == '__main__':
    opt = parser()
    setup_logger()
    model = torchvision.models.efficientnet_v2_m(weights=torchvision.models.EfficientNet_V2_M_Weights)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3, inplace=True),
        torch.nn.Linear(in_features=1280, out_features=2, bias=True)
    )
    real_dataset = loader.RealDataset(opt.data_root)
    fake_dataset = loader.FakeDataset(opt.data_root)
    real_loader = loader.Loader(real_dataset, opt.batch_size, opt.workers, True)
    fake_loader = loader.Loader(fake_dataset, opt.batch_size, opt.workers, True)
    patience = opt.patience
    optimizer = torch.optim.Adam(params=list(model.parameters()), lr=opt.lr, weight_decay=1e-5)
    
    for epoch in range(1, opt.epochs):
        for real, fake in zip(real_loader, fake_loader):
            pass