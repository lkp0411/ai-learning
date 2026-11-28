#只管杂活，比如作图和保存
import matplotlib.pyplot as plt
import torch

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def save_model(model,path = "linear.pt"):
    torch.save(model.states_dict(),path)

def plot_loss(train_loss,test_loss,lr,path = "loss_curve.png"):
    plt.figure(figsize=(10,6))
    plt.plot(train_loss,label = "验证集 Loss",color = "blue")
    plt.plot(test_loss,label = "验证集 Loss",color = "orange")
    plt.title(f"训练与验证损失曲线（LR = {lr}）")
    plt.legend()
    plt.grid(True,alpha = 0.3)
    plt.savefig(path,dpi = 150)
    plt.close()
