from neural_compressor.quantization import fit as fit
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.metric import METRICS
import torch
from torch.utils.data import DataLoader
from dataset.custom_deepfake import Custom_Dataset
import os
from model.network.Recce import Recce
import numpy as np
from time import time


# https://pytorch.org/tutorials/recipes/intel_neural_compressor_for_pytorch.html


def validate(val_loader, model):
    acc_meter = Acc()
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(len(val_loader), batch_time, prefix='Test: ')
    
    model.eval()
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(val_loader):
            start = time()
            imgs = imgs.cuda()
            pre = model(imgs)
            pre = torch.sigmoid(pre)
            pre = pre.cpu()
            pre = pre.squeeze()
            pre = torch.where(pre > 0.5, torch.tensor(1), torch.tensor(0))
            acc_meter.update(pre, labels)
            batch_time.update(time() - start)
            progress.print(i)

    acc = acc_meter.result()
    print(f"Batch Processing Time: {batch_time.sum}")
    print(f"Batch Accuracy: {acc}")
    return acc


def eval_func(model):
    acc = validate(val_loader, model)
    return float(acc)


class Acc(object):  
    def __init__(self, *args):  
        self.pred_list = []  
        self.label_list = []  
        self.samples = 0    

    def update(self, predict, label):  
        self.pred_list.extend(predict)  
        self.label_list.extend(label)  
        self.samples += len(label)    

    def reset(self):  
        self.pred_list = []  
        self.label_list = []  
        self.samples = 0    

    def result(self):  
        correct_num = np.sum(  np.array(self.pred_list) == np.array(self.label_list))  
        return correct_num / self.samples


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    

class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    

if __name__ == '__main__':
    # Loading fp32 model
    save_path = 'quantized_weight/'
    os.makedirs(save_path, exist_ok=True)
    fp32_model = Recce(num_classes=1)
    origin_weight_path = 'logs/lightning_logs/version_4/checkpoints/epoch=03-valid/AUROC=1.0000.ckpt'
    before_model_weight = torch.load(origin_weight_path)['state_dict']
    from collections import OrderedDict
    new_model_weight = OrderedDict()
    for key, val in before_model_weight.items():
        new_model_weight[key.replace('model.', '')] = val
    fp32_model.load_state_dict(new_model_weight, strict=True)
    fp32_model.cuda()
    fp32_model.eval()
    
    # Loader DataLoader
    val_loader = DataLoader(
        Custom_Dataset(root_path='../data/deepfake_classification/val'),
        batch_size=256,
        num_workers=8,
        pin_memory=True
    )
    
    # Loading Quantization
    conf = PostTrainingQuantConfig(
        approach='weight_only'
    )
    
    q_model = fit(
        model=fp32_model,
        conf=conf,
        calib_dataloader=val_loader,
        eval_func=eval_func,
    )
    
    # eval_func
    q_model.save(save_path)