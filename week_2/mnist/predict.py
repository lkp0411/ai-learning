import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import build_model

def predict():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型结构，并把 weights 灌进去
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load("best.pt"))
    model.eval()

    # 2. 随便拿一张测试集的图
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root='./data',train=False,download=True,transform=transform)
    image,true_label = test_dataset[0]

    # 把单张图形状变成 [1, 1, 28, 28] (Batch=1)
    image = image.unsqueeze(0).to(DEVICE)

    # 3. 预测
    with torch.no_grad():
        outputs = model(image)
        # 用 softmax 把输出转成概率（加起来等于 1）
        probabilities = F.softmax(outputs, dim=1)[0]

     # 4. 打印结果
    print(f"真实标签: {true_label}")
    for i in range(10):
        print(f"预测为数字 {i} 的概率: {probabilities[i].item()*100:.2f}%")

if __name__ == "__main__":
    predict()