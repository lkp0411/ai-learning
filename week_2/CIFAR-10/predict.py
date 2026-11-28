import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import build_model

# CIFAR-10 的真实类别名
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

def predict():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    model = build_model().to(DEVICE)
    # 加上 map_location，防止在没有GPU的电脑上报错
    model.load_state_dict(torch.load("best.pt", map_location=DEVICE))
    model.eval()

    # 2. 拿一张测试图（🔴 修复：必须加上和训练时一样的 Normalize）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # 取第 0 张图测试（你也可以改索引，比如测试 test_dataset[100]）
    image, true_label = test_dataset[0]

    # 🔴 修复：形状变成 [1, 3, 32, 32] (Batch=1, 3通道, 32x32)
    image = image.unsqueeze(0).to(DEVICE)

    # 3. 预测
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)[0]

    # 4. 打印结果
    print(f"真实标签: {true_label} ({CLASSES[true_label]})")
    print("-" * 30)
    
    # 为了好看，我们按概率从高到低排序
    top_prob, top_idx = torch.topk(probabilities, 3)
    
    print("预测概率最高的前3个类别：")
    for i in range(3):
        prob = top_prob[i].item() * 100
        idx = top_idx[i].item()
        print(f"{CLASSES[idx]:<12}: {prob:.2f}%")

if __name__ == "__main__":
    predict()