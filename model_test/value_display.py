import os
import torch
from matplotlib import pyplot as plt


# 功能描述：该函数输入两个参数 net类型的module 和 dataset类型的test_loader
# 将模型预测值与真实值用可视化图表的方式展示出来

def value_display(model, test_loader):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    true_y = []
    pre_y = []
    for X, y in test_loader:
        X = X.to(device)
        y = y.to(device)
        y_hat = model(X).detach().cpu().numpy()
        print(y_hat)
        y = y.to('cpu')
        for element in y:
            true_y.append(element)
        for element in y_hat:
            pre_y.append(element)
    # 绘制函数图像
    plt.plot([i for i in range(len(true_y))], true_y, label='ture_y')
    plt.plot([i for i in range(len(pre_y))], pre_y, label='pre_y')

    # 设置图像属性

    plt.title('Plot of True Values and Predicted Values')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()

    # 显示图像
    plt.show()


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    plt.title('Plot of True Values and Predicted Values')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
