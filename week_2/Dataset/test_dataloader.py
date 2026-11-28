import torch
from torch.utils.data import DataLoader          # ← 原来没导入，第 14 行会 NameError
from torchvision import datasets, transforms
from my_dataset import MyMNIST

# windows系统使用多进程必须放在 if __name__ == "__main__": 下
if __name__ == "__main__":
    print("正在准备数据集")
    # 1.实例化我的dataset
    train_dataset = MyMNIST(train=True)

    # 2. 扔进 DataLoader
    # num_workers=0 表示在主进程里读数据，最稳当（不会报错）
    # 等你测试成功了，可以把它改成 2 体验一下多进程
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,#打乱顺序
        num_workers=0,
    )

    # 3.测试一个batch
    print("开始读取数据...")
    for images, labels in train_loader:
        # 打印形状
        print(f"一个 Batch 的图片形状: {images.shape}")
        print(f"一个 Batch 的标签形状: {labels.shape}")

        # 看看里面的数据长什么样（比如看看图片的数值范围）
        print(f"图片数据的最大值: {images.max().item():.4f}，最小值: {images.min().item():.4f}")
        print(f"这个 batch 里的前 10 个标签: {labels[:10].tolist()}")

        # 只取第一个 batch 看看，取完就退出循环
        break

    print("测试完成！")
