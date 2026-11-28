import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --- 超参数集中配置 ---
EPOCHS = 100
LR = 0.002       # 原来 0.1 / 0.01 都太大：X 在 ±30 量级，步子一大就发散
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备:{device}")

# 解决 matplotlib 中文显示问题（第一周踩过的坑）
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

X = torch.randn(200, 1) * 10

y = 3 * X + 2 + torch.randn(200, 1) * 2

# 分出训练集和测试集
train_size = int(0.8 * len(X))
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)


def train():
    model = nn.Linear(1, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=LR)

    train_losses = []  # 建立一个空列表，用于存储每轮的训练集损失
    test_losses = []   # 建立一个空列表，用于存储每轮的验证集损失

    for epoch in range(EPOCHS):

        model.train()  # 显式声明训练模式（虽然线性回归没 BN 和 Dropout，但这是工程规范）

        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()  # 把上一步的梯度清零，不写这句会导致梯度累积爆炸

        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        train_losses.append(loss.item())

        # ============ 验证阶段 ==============
        model.eval()  # 切换到评估模式
        # 关闭梯度计算：验证阶段不需要反向传播
        with torch.no_grad():
            y_test_pred = model(X_test)
            test_loss = criterion(y_test_pred, y_test)
            test_losses.append(test_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch{epoch + 1:3d}|训练损失:{loss.item():.4f}|验证损失:{test_loss.item():.4f}")

    return model, train_losses, test_losses   # ← 必须在循环外面返回


def verify_model():
    # 验收要求：加载存下来的权重，测试推理
    print("\n---加载模型进行测试---")
    # 重新定义一个空模型结构，然后加载参数
    loaded_model = nn.Linear(1, 1).to(device)
    loaded_model.load_state_dict(torch.load("linear.pt"))
    loaded_model.eval()

    # 造一个测试数据 x = 10，理论上 y 应该在 32 左右
    test_x = torch.tensor([[10.0]]).to(device)
    with torch.no_grad():
        pred = loaded_model(test_x)

    print(f"输入X = 10.0，预测 Y = {pred.item():.4f}(真实理论值应接近32.0)")


if __name__ == "__main__":
    model, train_losses, test_losses = train()

    # ---保存模型---
    # 只保存参数字典(state_dict)，不保存整个模型结构，这样加载时不依赖源代码的类定义
    torch.save(model.state_dict(), "linear.pt")
    print("\n模型已保存至 linear.pt")

    print(f"学到的参数：w={model.weight.item():.4f}，b={model.bias.item():.4f}（真实 w=3，b=2）")

    # ---作图---
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="训练集 Loss", color="blue")
    plt.plot(test_losses, label="验证集 Loss", color="orange", linestyle="--")
    plt.title(f"训练与验证损失曲线(LR={LR})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("loss_curve.png", dpi=150)
    print("损失曲线已保存至 loss_curve.png")

    verify_model()
