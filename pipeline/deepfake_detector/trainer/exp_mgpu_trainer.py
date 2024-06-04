import os
import torch
import numpy as np
from model import load_model
from trainer import AbstractTrainer
from logging import info
import lightning as L
from custom_utils import Custom_Loader, Model
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

class ExpMultiGpuTrainer(AbstractTrainer):
    def __init__(self, config, stage="Train"):
        super(ExpMultiGpuTrainer, self).__init__(config, stage)
        np.random.seed(2021)

    def _train_settings(self, model_cfg, data_cfg, config_cfg):
        self.best_auc = 0
        self.best_acc = 0
        
        # load model
        self.num_classes = model_cfg["num_classes"]
        self.device = config_cfg['device']
        # print(model_cfg)  # {'num_classes': 1}
        self.model = load_model(self.model_name)(**model_cfg)
        self.model = Model(self.model, config_cfg)
        self.loader = Custom_Loader(root_path='../data/deepfake_classification/', data_cfg=data_cfg)
        logger = CSVLogger("logs")
        checkpoint_callback = ModelCheckpoint(
                            save_top_k=1,
                            monitor='valid/AUROC',
                            mode='max',
                            filename='{epoch:02d}-{valid/AUROC:.4f}')
        trainer = L.Trainer(strategy='ddp_find_unused_parameters_true',
                            accelerator='gpu', 
                            devices=4, 
                            sync_batchnorm=True,
                            check_val_every_n_epoch=1,
                            max_epochs=config_cfg['epoch'],
                            logger=logger,
                            callbacks=[checkpoint_callback],
                            log_every_n_steps=1)
        trainer.fit(self.model, 
                    datamodule=self.loader,)
        trainer.test(ckpt_path='best', 
                    datamodule=self.loader, 
                    verbose=True)

    def _save_pth(self, name):
        save_dir = os.path.join(self.dir, f"best_model_{name}.pth" )
        torch.save(self.model.state_dict(), save_dir)

    def _initiated_settings(self, model_cfg, data_cfg, config_cfg):
        pass
    def _test_settings(self, model_cfg, data_cfg, config_cfg):
        pass
    def _save_ckpt(self, step, best=False):
        pass
    def _load_ckpt(self, best=False, train=False):
        pass
    def train(self):
        pass
    def validate(self, epoch, step, timer, writer):
        pass
    def test(self):
        pass