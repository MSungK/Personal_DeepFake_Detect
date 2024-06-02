import torch


if __name__ == '__main__':
    path = 'logs/lightning_logs/version_3/checkpoints/epoch=00-valid/AUROC=1.0000.ckpt'
    model = torch.load(path)