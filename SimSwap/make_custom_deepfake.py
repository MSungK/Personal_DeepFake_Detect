#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: train.py
# Created Date: Monday December 27th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Friday, 22nd April 2022 10:49:26 am
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################

import os
from tqdm import tqdm
import random
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.backends import cudnn
import torch.utils.tensorboard as tensorboard

from util import util
from util.plot import plot_batch

from models.projected_model import fsModel
from data.data_loader_Swapping import GetLoader
from custom_utils import setup_logger, K_DataLoader
import logging
from PIL import Image


def str2bool(v):
    return v.lower() in ('true')

class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        
    def initialize(self):
        self.parser.add_argument('--name', type=str, default='simswap', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', default='0')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--isTrain', type=str2bool, default='True')

        # input/output sizes       
        self.parser.add_argument('--batchSize', type=int, default=16, help='input batch size')       

        # for displays
        self.parser.add_argument('--use_tensorboard', type=str2bool, default='False')

        # for training
        self.parser.add_argument('--dataset', type=str, default="/path/to/VGGFace2", help='path to the face swapping dataset')
        self.parser.add_argument('--continue_train', type=str2bool, default='False', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='./checkpoints/simswap224_test', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='10000', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=10000, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=10000, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0004, help='initial learning rate for adam')
        self.parser.add_argument('--Gdeep', type=str2bool, default='False')

        # for discriminators         
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add_argument('--lambda_id', type=float, default=30.0, help='weight for id loss')
        self.parser.add_argument('--lambda_rec', type=float, default=10.0, help='weight for reconstruction loss') 

        self.parser.add_argument("--Arc_path", type=str, default='arcface_model/arcface_checkpoint.tar', help="run ONNX model via TRT")
        self.parser.add_argument("--total_step", type=int, default=1000000, help='total training step')
        self.parser.add_argument("--log_frep", type=int, default=200, help='frequence for printing log information')
        self.parser.add_argument("--sample_freq", type=int, default=10000, help='frequence for sampling')
        self.parser.add_argument("--model_freq", type=int, default=10000, help='frequence for saving the model')
        self.isTrain = True
        
        # for Customizing TODO
        self.parser.add_argument('--custom', action='store_true', help='use customizing?')
        self.parser.add_argument("--model_path", type=str, required=True, help='path for loading model')
        # self.parser.add_argument("--G_path", type=str, default="", help='pre-trained Generator path')
        # self.parser.add_argument("--D_path", type=sstr, default="", help='pre-trained Discriminator path')
        
    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        if self.opt.isTrain:
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            util.mkdirs(expr_dir)
            if save and not self.opt.continue_train:
                file_name = os.path.join(expr_dir, 'opt.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k), str(v)))
                    opt_file.write('-------------- End ----------------\n')
        return self.opt


if __name__ == '__main__':
    setup_logger()
    opt         = TrainOptions().parse()
    iter_path   = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

    sample_path = os.path.join(opt.checkpoints_dir, opt.name, 'samples')

    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
    
    cudnn.benchmark = True

    model = fsModel()
    model.initialize(opt)

    imagenet_std    = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)
    imagenet_mean   = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)

    from random import randrange
    seed = randrange(1, 1000000)
    if opt.custom:
        train_loader = K_DataLoader(opt.dataset, opt.batchSize, 16, seed)
    else:
        train_loader    = GetLoader(opt.dataset, opt.batchSize, 16, seed)

    # Model Initialization
    logging.info(f"Model's path : {opt.model_path}")
    save_name = opt.model_path[opt.model_path.find('/')+1 : opt.model_path.find('_')]
    logging.info(f'save_name : {save_name}')
    
    before_model = torch.load(opt.model_path)
    from collections import OrderedDict
    new_dict = OrderedDict()
    for k, v in before_model.items():
        new_dict[f'netG.{k}'] = v
    model.netG.load_state_dict(before_model, strict=True)
    
    num_step = len(os.listdir(opt.dataset)) // opt.batchSize
    logging.info(f'num_step : {num_step}')
    
    model.netG.eval()
    cnt = 1
    
    for step in tqdm(range(num_step)):
        src_image1, src_image2  = train_loader.next()
        with torch.no_grad():
            arcface_112     = F.interpolate(src_image2,size=(112,112), mode='bicubic')
            id_vector_src2  = model.netArc(arcface_112)
            id_vector_src2  = F.normalize(id_vector_src2, p=2, dim=1)
            img_fake    = model.netG(src_image1, id_vector_src2).cpu()
                
            img_fake    = img_fake * imagenet_std
            img_fake    = img_fake + imagenet_mean
            img_fake    = img_fake.numpy().transpose(0,2,3,1)

            def postprocess(x):
                """[0,1] to uint8."""

                x = np.clip(255 * x, 0, 255)
                x = np.cast[np.uint8](x)
                return x
            
            img_fake = postprocess(img_fake)
            
            for i in range(img_fake.shape[0]):
                img = img_fake[i]
                Image.fromarray(img).save(f'../data/fake_images/{save_name}_{cnt}.png')
                cnt+=1