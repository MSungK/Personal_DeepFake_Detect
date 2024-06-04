import logging
import pytorch_lightning as pl
import lightning as L
from dataset.custom_deepfake import Custom_Dataset
from torch.utils import data
import torch
from os import path as osp
from optimizer import get_optimizer
from scheduler import get_scheduler
from loss import get_loss
from trainer.utils import MODELS_PATH, AccMeter, AUCMeter, AverageMeter, Logger, Timer
from trainer.utils import exp_recons_loss, MLLoss, reduce_tensor, center_print
from pprint import pprint
from torchmetrics.classification import Accuracy, AUROC
from torchmetrics.aggregation import MeanMetric, CatMetric
import torch.nn as nn

    
class Custom_Loader(L.LightningDataModule):
    def __init__(self, root_path, data_cfg):
        super().__init__()

        self.data_cfg = data_cfg
        self.train_dataset = Custom_Dataset(root_path=osp.join(root_path, 'train'))
        self.valid_dataset = Custom_Dataset(root_path=osp.join(root_path, 'val'))
        self.test_dataset = Custom_Dataset(root_path=osp.join(root_path, 'test'))
        self.prepare_data_per_node = True
    
    # def prepare_data(self):
    #     pass # TODO

    # def setup(self, stage=None):
    #     pass # TODO
        
    def train_dataloader(self):
        self.train_loader = data.DataLoader(self.train_dataset, shuffle=True,
                                            num_workers=self.data_cfg.get("num_workers", 8),
                                            batch_size=self.data_cfg["train_batch_size"],
                                            pin_memory=True,
                                            drop_last=True)
        return self.train_loader
    
    def val_dataloader(self):
        self.val_loader = data.DataLoader(self.valid_dataset, shuffle=False,
                                        num_workers=self.data_cfg.get("num_workers", 8),
                                        batch_size=self.data_cfg["val_batch_size"],
                                        pin_memory=True)
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = data.DataLoader(self.test_dataset, shuffle=False,
                                        num_workers=self.data_cfg.get("num_workers", 8),
                                        batch_size=self.data_cfg["val_batch_size"],
                                        pin_memory=True)
        return self.test_loader
        


class Model(L.LightningModule):
    def __init__(self, model, config_cfg):
        super().__init__()
        self.model = model
        self.config_cfg = config_cfg
        self.optim_cfg = config_cfg['optimizer']
        
    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.optim_cfg.pop("name"))(self.model.parameters(), **self.optim_cfg)
        # load scheduler
        self.scheduler = get_scheduler(self.optimizer, self.config_cfg.get("scheduler", None))
        # load loss
        # TODO
        self.loss_criterion = get_loss(self.config_cfg.get("loss", None), device='cpu') 
        self.num_epoch = self.config_cfg['epoch']

        # balance coefficients
        self.lambda_1 = self.config_cfg["lambda_1"]
        self.lambda_2 = self.config_cfg["lambda_2"]
        self.warmup_step = self.config_cfg.get('warmup_step', 0)

        self.contra_loss = MLLoss()
        self.metric = nn.ModuleDict({
            'train_acc_meter' : Accuracy(task='binary').to(self.device),
            'valid_acc_meter' : Accuracy(task='binary').to(self.device),
            'train_auc_meter' : AUROC(task='binary').to(self.device),
            'valid_auc_meter' : AUROC(task='binary').to(self.device),
            'train_loss_meter' : MeanMetric().to(self.device),
            'valid_loss_meter' : MeanMetric().to(self.device),
            'train_recons_loss_meter' : MeanMetric().to(self.device),
            'valid_recons_loss_meter' : MeanMetric().to(self.device),
            'train_contra_loss_meter' : MeanMetric().to(self.device),
            'valid_contra_loss_meter' : MeanMetric().to(self.device),
            'test_acc_meter' : Accuracy(task='binary').to(self.device),
            'test_auc_meter' : AUROC(task='binary').to(self.device),
        })
        return self.optimizer
    
    def training_step(self, batch, batch_idx):
        imgs, Y = batch
        if self.warmup_step != 0 and self.current_epoch <= self.warmup_step:
            lr = self.config['config']['optimizer']['lr'] * float(self.current_epoch) / self.warmup_step
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        Y_pre = self.model(imgs)
        # for BCE Setting:
        Y_pre = Y_pre.squeeze()
        loss = self.loss_criterion(Y_pre, Y.float())
        Y_pre = torch.sigmoid(Y_pre)

        loss = (loss - 0.04).abs() + 0.04
        recons_loss = exp_recons_loss(self.model.loss_inputs['recons'], (imgs, Y))
        contra_loss = self.contra_loss(self.model.loss_inputs['contra'], Y)
        loss += self.lambda_1 * recons_loss + self.lambda_2 * contra_loss
        
        if self.warmup_step == 0 or self.current_epoch > self.warmup_step:
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()
        
        self.metric['train_acc_meter'].update(Y_pre.detach(), Y.detach())
        self.metric['train_auc_meter'].update(Y_pre.detach(), Y.detach())
        self.metric['train_loss_meter'].update(loss.detach())
        self.metric['train_recons_loss_meter'].update(recons_loss.detach())
        self.metric['train_contra_loss_meter'].update(contra_loss.detach())
        
        self.log('step_loss', loss, sync_dist=True, on_step=True)
        return loss

    def on_train_epoch_end(self):
        auc = self.metric['train_acc_meter'].compute()
        acc = self.metric['train_auc_meter'].compute()
        recons_loss = self.metric['train_loss_meter'].compute()
        contra_loss = self.metric['train_recons_loss_meter'].compute()
        loss = self.metric['train_contra_loss_meter'].compute()
        
        lr = self.scheduler.get_last_lr()[0]
    
        self.log("train/AUROC",auc, sync_dist=True)
        self.log("train/Acc",acc, sync_dist=True)
        self.log('train/Loss',loss, sync_dist=True)
        self.log("train/Recons_Loss",recons_loss, sync_dist=True)
        self.log("train/Contra_Loss",contra_loss, sync_dist=True)
        self.log("train/LR",lr, sync_dist=True)
        
        self.metric['train_acc_meter'].reset()
        self.metric['train_auc_meter'].reset()
        self.metric['train_loss_meter'].reset()
        self.metric['train_recons_loss_meter'].reset()
        self.metric['train_contra_loss_meter'].reset()
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        imgs, Y = batch
        Y_pre = self.model(imgs)
        # for BCE Setting:
        Y_pre = Y_pre.squeeze()
        loss = self.loss_criterion(Y_pre, Y.float())
        Y_pre = torch.sigmoid(Y_pre)
        
        loss = (loss - 0.04).abs() + 0.04
        recons_loss = exp_recons_loss(self.model.loss_inputs['recons'], (imgs, Y))
        contra_loss = self.contra_loss(self.model.loss_inputs['contra'], Y)
        loss += self.lambda_1 * recons_loss + self.lambda_2 * contra_loss
        
        self.metric['valid_acc_meter'].update(Y_pre, Y)
        self.metric['valid_auc_meter'].update(Y_pre, Y)
        self.metric['valid_loss_meter'].update(loss)
        self.metric['valid_recons_loss_meter'].update(recons_loss)
        self.metric['valid_contra_loss_meter'].update(contra_loss)
        
        return loss
    
    def on_validation_epoch_end(self):
        auc = self.metric['valid_acc_meter'].compute()
        acc = self.metric['valid_auc_meter'].compute()
        recons_loss = self.metric['valid_loss_meter'].compute()
        contra_loss = self.metric['valid_recons_loss_meter'].compute()
        loss = self.metric['valid_contra_loss_meter'].compute()
        
        self.log("valid/AUROC",auc, sync_dist=True)
        self.log("valid/Acc",acc, sync_dist=True)
        self.log('valid/Loss',loss, sync_dist=True)
        self.log("valid/Recons_Loss",recons_loss, sync_dist=True)
        self.log("valid/Contra_Loss",contra_loss, sync_dist=True)

        self.metric['valid_acc_meter'].reset()
        self.metric['valid_auc_meter'].reset()
        self.metric['valid_loss_meter'].reset()
        self.metric['valid_recons_loss_meter'].reset()
        self.metric['valid_contra_loss_meter'].reset()
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.model.eval()
        imgs, Y = batch
        Y_pre = self.model(imgs)
        # for BCE Setting:
        Y_pre = Y_pre.squeeze()
        loss = self.loss_criterion(Y_pre, Y.float())
        Y_pre = torch.sigmoid(Y_pre)

        self.metric['test_acc_meter'].update(Y_pre, Y)
        self.metric['test_auc_meter'].update(Y_pre, Y)
    
    def on_test_epoch_end(self):
        auc = self.metric['test_acc_meter'].compute()
        acc = self.metric['test_auc_meter'].compute()
        self.log("test/AUROC",auc, sync_dist=True)
        self.log("test/Acc",acc, sync_dist=True)

        auc = self.metric['test_acc_meter'].reset()
        acc = self.metric['test_auc_meter'].reset()