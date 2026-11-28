import torch.nn as nn

def build_model():
    #线性回归：输入一维，输出一维
    return nn.Linear(1,1)