import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchvision import datasets,transforms
from model import build_model


#配置
DEVICE = torch.device("cuda"if torch.cuda.is_available() else"cpu")
EPOCHS = 20
BATCH_SIZE = 64
LR = 0.001
PATIENCE = 3  #早停：连续 3 轮验证集准确率不提升，就停止训练

#训练集的变换（带数据增强）
train_transform = transforms.Compose([
    transforms.RandomCrop(32,padding=4),#随机裁剪，四周补零
    transforms.RandomHorizontalFlip(),#随机翻转
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))#归一化
])

#验证集的变化
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

def train():
    print(f"使用设备: {DEVICE}")
    #初始化
    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()#多分类必须使用交叉熵
    optimizer = optim.Adam(model.parameters(),lr=LR)

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        for images,labels in train_loader:
            images,labels = images.to(DEVICE),labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

        #==========验证阶段==========
        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for images,labels in val_loader:
                images,labels = images.to(DEVICE),labels.to(DEVICE)

                outputs = model(images)
                val_loss += criterion(outputs,labels).item()

                #算出预测的类别，并对比真实标签
                _,predicted = torch.max(outputs.data,1)#在10个打分中找出得分最高的那个类别的编号
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct/total
        avg_val_loss = val_loss/len(val_loader)

        print(f"Epoch {epoch+1:02d} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        # ================= 存 Best & 早停 =================
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(),"best.pt")
            print(f"  --> 发现新高！保存 best.pt (Acc: {best_acc*100:.2f}%)")
            patience_counter =0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"连续 {PATIENCE} 轮未提升，触发早停！")
                break

    total_time = time.time() - start_time
    print("\n========= 训练总结 =========")
    print(f"总用时: {total_time:.1f} 秒")
    print(f"最佳轮次: Epoch {best_epoch}")
    print(f"最佳验证准确率: {best_acc*100:.2f}%")

if __name__ == "__main__":
    train()
        