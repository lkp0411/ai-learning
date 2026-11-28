import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
LR = 0.1
TRAIN_RATIO = 0.8
BATCH_SIZE = 32