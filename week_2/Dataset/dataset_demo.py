import torch
from torchvision import datasets, transforms

#transform负责把图片转化为tensor
transform = transforms.ToTensor()

#下载并加载MNIST数据集
train_data = datasets.MNIST(root='./data',train=True,transform=transform,download=True)
print(f"总共有多少张图：{len(train_data)}")

#取第0张图查看
img,label = train_data[0]
print(f"单张图片的形状: {img.shape}") # 应该是 torch.Size([1, 28, 28])
print(f"标签: {label}")