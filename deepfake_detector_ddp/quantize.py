from neural_compressor.quantization import fit as fit
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion
from model import load_model
import torch
from torch.utils.data import DataLoader
from dataset.custom_deepfake import Custom_Dataset
import os
from model.network.Recce import Recce
from model import load_model
import numpy as np

class Acc(object):  
    def __init__(self, *args):  
        self.pred_list = []  
        self.label_list = []  
        self.samples = 0    

    def update(self, predict, label):  
        self.pred_list.extend(np.argmax(predict, axis=1))  
        self.label_list.extend(label)  
        self.samples += len(label)    

    def reset(self):  
        self.pred_list = []  
        self.label_list = []  
        self.samples = 0    

    def result(self):  
        correct_num = np.sum(  np.array(self.pred_list) == np.array(self.label_list))  
        return correct_num / self.samples


if __name__ == '__main__':
    save_path = 'quantized_weight/'
    os.makedirs(save_path, exist_ok=True)
    origin_model = Recce(num_classes=1)
    origin_weight_path = 'logs/lightning_logs/version_4/checkpoints/epoch=03-valid/AUROC=1.0000.ckpt'
    before_model_weight = torch.load(origin_weight_path)['state_dict']
    from collections import OrderedDict
    new_model_weight = OrderedDict()
    for key, val in before_model_weight.items():
        new_model_weight[key.replace('model.', '')] = val
        
    origin_model.load_state_dict(new_model_weight, strict=True)
    origin_model.eval()
    
    val_loader = DataLoader(
        Custom_Dataset(root_path='../data/deepfake_classification/val'),
        batch_size=256,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
    )
    
    tuning_criterion = TuningCriterion(max_trials=600)
    conf = PostTrainingQuantConfig(
        tuning_criterion=tuning_criterion
    )
    q_model = fit(
        model=origin_model,
        conf=conf,
        calib_dataloader=val_loader,
        eval_dataloader=val_loader,
        eval_metric=Acc()
    )
    # eval_func
    q_model.save(save_path)
    
    # for img, label in val_loader:
    #     y_pre = origin_model(img)
    #     y_pre = torch.sigmoid(y_pre)
    #     y_pre = y_pre.squeeze()
    #     y_pre = torch.where(y_pre > 0.5, torch.tensor(1), torch.tensor(0))
    #     print(torch.sum(y_pre==label))
    #     print(y_pre.shape)
    #     exit()