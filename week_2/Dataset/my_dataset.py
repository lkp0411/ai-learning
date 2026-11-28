import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MyMNIST(Dataset):
    def __init__(self, train=True):
        # 内部还是借用官方的把数据下载下来
        self.data = datasets.MNIST(
            root='./data', train=train, download=True, transform=transforms.ToTensor()
        )

    # ↓ 这两个方法必须在 class 层级（缩进 4 格），不能缩进到 __init__ 里面
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # DataLoader 每次会传一个 idx 进来，需要返回对应的 x 和 y
        img, label = self.data[idx]
        return img, label
