import os
from glob import glob
import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
import logging
from custom_utils import setup_logger


def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transformer_Arcface = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

detransformer = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
    ])
if __name__ == '__main__':
    opt = TestOptions().parse() 
    setup_logger()
    
    start_epoch, epoch_iter = 1, 0

    torch.nn.Module.dump_patches = True
    model = create_model(opt)
    model.eval()
    path = 'checkpoints/asian_face/5000_net_G.pth'
    # path = 'checkpoints/people/latest_net_G.pth'
    
    before_model = torch.load(path)
    
    # f = open('memo.txt', 'w')
    # for k, v in model.state_dict().items():
    #     f.write(k + '\n')
    # exit()
    # print('===' * 10)
    from collections import OrderedDict
    new_dict = OrderedDict()
    for k, v in before_model.items():
        new_dict[f'netG.{k}'] = v
    # for k, v in new_dict.items():
    #     f.write(k + '\n')
    model.netG.load_state_dict(before_model, strict=True)
    source_path = opt.source_path
    target_path = opt.target_path
    
    for source in glob(source_path + '/*'):
        for target in glob(target_path + '/*'):
            target = transformer_Arcface(Image.open(target)).unsqueeze(0).cuda()
            source = transformer_Arcface(Image.open(source)).unsqueeze(0).cuda()

            with torch.no_grad():
                arcface_112     = F.interpolate(target,size=(112,112), mode='bicubic')
                id_vector_src1  = model.netArc(arcface_112)
                id_vector_src1  = F.normalize(id_vector_src1, p=2, dim=1)
                img_fake    = model.netG(source, id_vector_src1).cpu()
                    
                img_fake    = detransformer(img_fake)
                
                img_fake    = img_fake.squeeze(0).numpy()
                img_fake = np.clip(255 * img_fake, 0, 255)
                img_fake = np.cast[np.uint8](img_fake)
                img_fake = np.transpose(img_fake, (1,2,0))
                # print(img_fake)
                # exit()
                logging.info(img_fake.shape)
                os.makedirs(opt.output_path, exist_ok=True)
                Image.fromarray(img_fake).save(opt.output_path + 'result.jpg')
                logging.info(opt.output_path + 'result.jpg')
                break