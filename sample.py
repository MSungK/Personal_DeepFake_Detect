import torch


if __name__ == '__main__':
    path = 'SimSwap/checkpoints/people/latest_net_D2.pth'
    model = torch.load(path)
    for key, value in model.items():
        print(f'{key} : {value.shape}')