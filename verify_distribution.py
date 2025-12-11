import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from model_test.grid_research import test_model
from data_processing.timeseries_dataset import loading_car_hacking, dataClassifier
from model.LSTM import LSTMAutoencoder
from NLT.NLT_main import likelihood_transformation


def verify_full_data():
    # 1. 设置参数
    window_size = 10
    sliding_window = 1
    input_size = 1
    hidden_size = 50
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(">>> 开始全量数据验证 (不再进行随机切分) <<<")

    # 2. 加载模型
    model = LSTMAutoencoder(input_size, hidden_size, device)
    try:
        model.load_state_dict(torch.load('car_hacking_model.pt'))
        model.to(device)
        model.eval()
        print("✅ 模型加载成功")
    except:
        print("❌ 未找到模型文件 car_hacking_model.pt")
        return

    # 3. 加载全量数据
    transfer = likelihood_transformation()
    # 加载所有数据
    full_dataset = loading_car_hacking(window_size, sliding_window, transfer, 'all')

    # 获取标签和数据加载器
    # 注意：这里我们手动分离标签，确保绝对对齐
    data_loader = DataLoader(full_dataset, batch_size=512, shuffle=False)

    All_Labels = []
    print("正在提取全量标签...")
    # 直接从 dataset 获取所有 label (0=Attack, 1=Normal)
    # TimeSeriesDataset.label 是一个 tensor
    All_Labels = full_dataset.label.numpy()

    # 4. 计算全量 Loss
    print("正在计算全量 Loss (这可能需要一点时间)...")
    criterion = nn.MSELoss(reduction='none')
    loss_ar = test_model(model, data_loader, criterion, device)

    # 5. 分离正常与攻击
    # 你的数据集中: 0=Attack, 1=Normal
    normal_losses = loss_ar[All_Labels == 1]
    attack_losses = loss_ar[All_Labels == 0]

    print(f"\n【正常样本 Normal】 (共 {len(normal_losses)} 条)")
    print(f"Min: {normal_losses.min():.6f}")
    print(f"Max: {normal_losses.max():.6f}")
    print(f"Mean: {normal_losses.mean():.6f}")

    print(f"\n【攻击样本 Attack】 (共 {len(attack_losses)} 条)")
    print(f"Min: {attack_losses.min():.6f}")
    print(f"Max: {attack_losses.max():.6f}")
    print(f"Mean: {attack_losses.mean():.6f}")

    # 6. 最终判定
    overlap_start = max(normal_losses.min(), attack_losses.min())
    overlap_end = min(normal_losses.max(), attack_losses.max())

    # 真正的分离条件：Max(Normal) < Min(Attack)
    # 或者 Min(Normal) > Max(Attack) (不太可能)

    if normal_losses.max() < attack_losses.min():
        print("\n🎉 恭喜！全量数据完全可分！模型完美！")
        print(f"安全阈值区间: ({normal_losses.max():.6f}, {attack_losses.min():.6f})")
    else:
        print("\n⚠️ 警告：全量数据存在重叠 (Overlap)")
        print(f"攻击样本最小值 ({attack_losses.min():.6f}) 低于 正常样本最大值 ({normal_losses.max():.6f})")
        print("这意味着：无论怎么选阈值，必然会有误报或漏报。")
        print("之前的 1.0 只是测试集划分时的运气。")

    # 7. 画图
    plt.figure(figsize=(12, 6))
    plt.hist(normal_losses, bins=100, alpha=0.6, label='Normal', color='green', log=True)
    plt.hist(attack_losses, bins=100, alpha=0.6, label='Attack', color='red', log=True)
    plt.title("Full Dataset Loss Distribution")
    plt.xlabel("MSE Loss")
    plt.ylabel("Count (Log Scale)")
    plt.legend()
    plt.savefig('full_distribution_check.png')
    print("全量分布图已保存为 full_distribution_check.png")


if __name__ == '__main__':
    verify_full_data()