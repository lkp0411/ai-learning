import torch.nn as nn


class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # 把 [64,1,28,28] 展平成 [64, 784]
        self.fc = nn.Sequential(  # 全连接网络
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),  # 输出 10 个类别（数字 0-9）
        )

    def forward(self, x):  # 前向传播
        x = self.flatten(x)
        return self.fc(x)


def build_model():
    return MnistNet()
