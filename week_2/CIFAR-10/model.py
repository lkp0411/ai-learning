import torch.nn as nn

class CifarCNN(nn.Module):
    def __init__(self):
        super().__init__()
        #卷积块1：输入三通道（RGB），输出32通道
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)#图片长宽减半
        )
        #卷积块2：输入32，输出64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        #卷积块3：输入64，输出128
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        #全连接层分类
        self.flatten = nn.Flatten()#全连接层只能接受一苇向量，所以要把多维矩阵拉成一维向量
        self.fc = nn.Sequential(
            nn.Linear(128*4*4,256),
            nn.ReLU(),
            nn.Dropout(0.5),#防过拟合
            nn.Linear(256,10)
        )
    """
    这是全连接层的输入维度。

    回想一下你的网络结构：

    输入图片是 32 x 32。
    经过第一个 MaxPool2d(2)（池化），长宽减半：16 x 16。
    经过第二个 MaxPool2d(2)，长宽再减半：8 x 8。
    经过第三个 MaxPool2d(2)，长宽最后一次减半：4 x 4。
    最后一个卷积层输出了 128 个通道。
    所以，数据到达这里时的形状是 [64, 128, 4, 4]。展平后，每个样本包含的特征数就是 128 × 4 × 4 = 2048。
    """

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return self.fc(x)

def build_model():
    return CifarCNN()