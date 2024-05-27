import os
import sys
import time
import math
import yaml
import torch
import random
import numpy as np

from tqdm import tqdm
from pprint import pprint
from torch.utils import data
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter

from dataset import load_dataset
from loss import get_loss
from model import load_model
from optimizer import get_optimizer
from scheduler import get_scheduler
from trainer import AbstractTrainer, LEGAL_METRIC
from trainer.utils import exp_recons_loss, MLLoss, reduce_tensor, center_print
from trainer.utils import MODELS_PATH, AccMeter, AUCMeter, AverageMeter, Logger, Timer
from logging import info
from dataset.custom_deepfake import Custom_Dataset

class ExpMultiGpuTrainer(AbstractTrainer):
    def __init__(self, config, stage="Train"):
        super(ExpMultiGpuTrainer, self).__init__(config, stage)
        np.random.seed(2021)

    def _mprint(self, content=""):
        if self.local_rank == 0:
            print(content)

    def _initiated_settings(self, model_cfg=None, data_cfg=None, config_cfg=None):
        pass

    def _train_settings(self, model_cfg, data_cfg, config_cfg):
        # debug mode: no log dir, no train_val operation.
        self.debug = config_cfg["debug"]
        self._mprint(f"Using debug mode: {self.debug}.")
        self._mprint("*" * 20)

        self.eval_metric = config_cfg["metric"]
        if self.eval_metric not in LEGAL_METRIC:
            raise ValueError(f"Evaluation metric must be in {LEGAL_METRIC}, but found " 
                             f"{self.eval_metric}.")
        if self.eval_metric == LEGAL_METRIC[-1]:
            self.best_metric = 1.0e8

        # load training dataset
        """
        train_dataset = data_cfg["file"]
        branch = data_cfg["train_branch"]
        name = data_cfg["name"]
        """
        self.train_set = Custom_Dataset(root_path='../data/deepfake_classification/train')
        
        # wrapped with data loader
        self.train_loader = data.DataLoader(self.train_set, shuffle=True,
                                            num_workers=data_cfg.get("num_workers", 8),
                                            batch_size=data_cfg["train_batch_size"],
                                            pin_memory=True,
                                            drop_last=True)
    
        if self.local_rank == 0:
            # load validation dataset
            self.val_set = Custom_Dataset(root_path='../data/deepfake_classification/val')
            # wrapped with data loader
            self.val_loader = data.DataLoader(self.val_set, shuffle=False,
                                            num_workers=data_cfg.get("num_workers", 16),
                                            batch_size=data_cfg["val_batch_size"],
                                            pin_memory=True)

        self.resume = config_cfg.get("resume", False) # TODO
        
        # load model
        self.num_classes = model_cfg["num_classes"]
        self.device = config_cfg['device']
        self.model = load_model(self.model_name)(**model_cfg).to(self.device)
        # self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
        # self._mprint(f"Using SyncBatchNorm.")
        # self.model = torch.nn.parallel.DistributedDataParallel(
        #     self.model, device_ids=[self.local_rank], find_unused_parameters=True)

        # load optimizer
        optim_cfg = config_cfg.get("optimizer", None)
        optim_name = optim_cfg.pop("name")
        self.optimizer = get_optimizer(optim_name)(self.model.parameters(), **optim_cfg)
        # load scheduler
        self.scheduler = get_scheduler(self.optimizer, config_cfg.get("scheduler", None))
        # load loss
        self.loss_criterion = get_loss(config_cfg.get("loss", None), device=self.device)

        # total number of steps (or epoch) to train
        # self.num_steps = train_options["num_steps"]
        
        self.num_epoch = config_cfg['epoch']

        # the number of steps to write down a log
        # self.log_steps = train_options["log_steps"]
        
        # the number of steps to validate on val dataset once
        # self.val_steps = train_options["val_steps"]

        # balance coefficients
        self.lambda_1 = config_cfg["lambda_1"]
        self.lambda_2 = config_cfg["lambda_2"]
        self.warmup_step = config_cfg.get('warmup_step', 0)

        self.contra_loss = MLLoss()
        self.acc_meter = AccMeter()
        self.auc_meter = AUCMeter()
        self.loss_meter = AverageMeter()
        self.recons_loss_meter = AverageMeter()
        self.contra_loss_meter = AverageMeter()
        self.log_path = config_cfg['log_path']

        # if self.resume and self.local_rank == 0:
        #     self._load_ckpt(best=config_cfg.get("resume_best", False), train=True)

    def _test_settings(self, model_cfg, data_cfg, config_cfg):
        # Not used.
        raise NotImplementedError("The function is not intended to be used here.")

    def _load_ckpt(self, best=False, train=False):
        # Not used.
        raise NotImplementedError("The function is not intended to be used here.")

    def _save_pth(self, name):
        save_dir = os.path.join(self.dir, f"best_model_{name}.pth" )
        torch.save(self.model.state_dict(), save_dir)

    def train(self):
        timer = Timer()
        grad_scalar = GradScaler(2 ** 10)
        
        cur_step = 1 
        """
        cur_step 증가 코드 넣어야 함
        """
        f = open(self.log_path, 'a')
        
        for epoch_idx in range(1, self.num_epoch + 1):
            # reset meter
            self.acc_meter.reset()
            self.auc_meter.reset()
            self.loss_meter.reset()
            self.recons_loss_meter.reset()
            self.contra_loss_meter.reset()
            self.optimizer.step()

            for batch_idx, train_data in tqdm(enumerate(self.train_loader)):
                self.model.train()
                imgs, Y = train_data
                imgs = imgs.to(self.device)
                Y = Y.to(self.device)

                # warm-up lr
                if self.warmup_step != 0 and cur_step <= self.warmup_step:
                    lr = self.config['config']['optimizer']['lr'] * float(cur_step) / self.warmup_step
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr

                self.optimizer.zero_grad()
                with autocast():
                    Y_pre = self.model(imgs)
                    # for BCE Setting:
                    if self.num_classes == 1:
                        Y_pre = Y_pre.squeeze()
                        loss = self.loss_criterion(Y_pre, Y.float())
                        Y_pre = torch.sigmoid(Y_pre)
                    else:
                        loss = self.loss_criterion(Y_pre, Y)

                    # flood
                    loss = (loss - 0.04).abs() + 0.04
                    recons_loss = exp_recons_loss(self.model.loss_inputs['recons'], (imgs, Y))
                    contra_loss = self.contra_loss(self.model.loss_inputs['contra'], Y)
                    loss += self.lambda_1 * recons_loss + self.lambda_2 * contra_loss

                grad_scalar.scale(loss).backward()
                grad_scalar.step(self.optimizer)
                grad_scalar.update()
                if self.warmup_step == 0 or cur_step > self.warmup_step:
                    self.scheduler.step()

                self.acc_meter.update(Y_pre, Y, self.num_classes == 1)
                self.auc_meter.update(Y_pre, Y, use_bce=True)
                self.loss_meter.update(loss.item())
                self.recons_loss_meter.update(recons_loss.item())
                self.contra_loss_meter.update(contra_loss.item())
                cur_step+=1
                info(f'Current Epoch : {epoch_idx} - {batch_idx+1}/{len(self.train_loader)} : loss : {loss.item()}')

            
            '''
            Current Epoch
            train/~
            valid/~
            '''
            f.write(f"Current Epoch : {epoch_idx}\n")
            f.write(f"train/AUROC: {self.auc_meter.mean_auc} ")
            f.write(f"train/Acc: {self.acc_meter.avg} train/Loss: {self.loss_meter.avg} ")
            f.write(f"train/Recons_Loss: {self.recons_loss_meter.avg} ")
            f.write(f"train/Contra_Loss: {self.contra_loss_meter.avg}")
            f.write(f"train/LR: {self.scheduler.get_last_lr()[0]}\n")

            # validating process
            self.validate(epoch_idx, f)
                

    def validate(self, epoch, f):
        self.model.eval()
        with torch.no_grad():
            self.acc_meter.reset()
            self.auc_meter.reset()
            self.loss_meter.reset()
            self.recons_loss_meter.reset()
            self.contra_loss_meter.reset()
            
            for data in tqdm(self.val_loader):
                imgs, Y = data
                imgs = imgs.to(self.device)
                Y = Y.to(self.device)
                Y_pre = self.model(imgs)

                # for BCE Setting:
                if self.num_classes == 1:
                    Y_pre = Y_pre.squeeze()
                    loss = self.loss_criterion(Y_pre, Y.float())
                    Y_pre = torch.sigmoid(Y_pre)
                else:
                    loss = self.loss_criterion(Y_pre, Y)
                # flood
                loss = (loss - 0.04).abs() + 0.04
                recons_loss = exp_recons_loss(self.model.loss_inputs['recons'], (imgs, Y))
                contra_loss = self.contra_loss(self.model.loss_inputs['contra'], Y)
                loss += self.lambda_1 * recons_loss + self.lambda_2 * contra_loss
                
                self.acc_meter.update(Y_pre, Y, self.num_classes == 1)
                self.auc_meter.update(Y_pre, Y, use_bce=True)
                self.loss_meter.update(loss.item())
                self.recons_loss_meter.update(recons_loss.item())
                self.contra_loss_meter.update(contra_loss.item())
            
            '''
            Current Epoch
            train/~
            valid/~
            '''
            cur_auc = self.auc_meter.mean_auc
            cur_acc = self.acc_meter.avg
            cur_loss = self.loss_meter.avg
            f.write(f"valid/AUROC: {cur_auc} ")
            f.write(f"valid/Acc: {cur_acc} train/Loss: {cur_loss} ")
            f.write(f"valid/Recons_Loss: {self.recons_loss_meter.avg} ")
            f.write(f"valid/Contra_Loss: {self.contra_loss_meter.avg}")

            if cur_auc > self.best_auc:
                self.best_auc = cur_auc
                self._save_pth('auc')
                
            if cur_acc > self.best_acc:
                self.best_acc = cur_acc
                self._save_pth('acc')

    def test(self):
        # Not used.
        raise NotImplementedError("The function is not intended to be used here.")
