import torch
from config import TRAIN_RATIO,DEVICE

def get_data():
    X = torch.randn(200,1)*10
    y = 3*X + 2 + torch.randn(200,1)*2

    train_size = int(TRAIN_RATIO * len(X))
    X_train,y_train = X[:train_size],y[:train_size].to(DEVICE)
    X_test,y_test = X[train_size:],y[train_size:].to(DEVICE)
    return X_train,y_train,X_test,y_test