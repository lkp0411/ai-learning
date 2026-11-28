import torch
import torch.nn as nn
import torch.optim as optim
from config import EPOCHS,LR,DEVICE
from data import get_data
from model import build_model
from utils import save_model,plot_loss

def train():
    X_train,y_train,X_test,y_test = get_data()

    model = build_model().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(),lr = LR)

    train_losses,test_losses = [],[]

    for epochs in range(EPOCHS):
        model.train()
        y_pred = model(X_train)
        loss = criterion(y_pred,X_train)
        optimizer.no_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred,X_test)
            test_losses.append(test_loss.item())

    save_model(model)
    plot_loss(train_losses,test_losses,LR)
    print("训练结束，模型和图片已保存。")

if __name__ == "__main__":
    train()
