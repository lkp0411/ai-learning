import torch
import torch.nn as nn
import torch.optim as optim

#检查是否使用gpu
device = torch.device("cuda" if torch.cuda.is_available()else "cpu" )
print(f"当前使用的设备:{device}")

#1.准备数据
#生成100个x，形状是(100,1)
X = torch.randn(100,1)*10#randn生成的是标准正态分布，乘以10是为了让数据更大一些

#真实参数：w=3，b=2
#y = 3x + 2 + 噪声
y = 3*X + 2 + torch.randn(100,1)*2

#把数据放到gpu上
X = X.to(device)
y = y.to(device)

#2.定义模型:最简单的线性层，输入一维，输出一维
model = nn.Linear(1,1).to(device)

#3.定义损失函数:均方误差（MSE）
criterion = nn.MSELoss()

#4.定义优化器:随机梯度下降（SGD）,学习率设为0.01
optimizer = optim.SGD(model.parameters(),lr=0.01)

#5.训练100轮
epochs = 100

for epoch in range(epochs):
    #---标准五步---

    #①前向传播：拿数据算预测值
    y_pred = model(X)

    #②计算损失：预测值和真实值差多少
    loss = criterion(y_pred,y)

    #③梯度清零：把上一步的梯度清零
    optimizer.zero_grad()

    #④反向传播：自动计算每个参数的梯度
    loss.backward()

    #⑤更新参数：根据梯度调整w和b
    optimizer.step()

    #打印每10轮的损失
    if(epoch + 1) % 10 == 0:
        print(f"第{epoch+1}轮，损失:{loss.item():.4f}")

#6.验证结果：把学到的w和b取出来
print("\n---训练完成---")
#model.weight和model.bias是Parameter类型，需要用item()转成普通浮点数
learned_w = model.weight.item()
learned_b = model.bias.item()

print(f"真实的参数：w=3.000，b=2.000")
print(f"学到的参数：w={learned_w:.4f}，b={learned_b:.4f}")