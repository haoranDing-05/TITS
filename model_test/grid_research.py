import torch
import numpy as np
import math
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def prepare_data_loaders(train_dataset):
    """
    从 Dataset 中分离出用于 Loss 计算的 Loader 和用于评估的 Label
    """
    # 增大batch_size加速推理
    train_loss_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)

    X = []
    Label = []
    print("正在提取测试集标签...")
    for i in range(len(train_dataset)):
        # get_test_sample 返回 (data, label)
        _, label = train_dataset.get_test_sample(i)
        Label.append(label)

    return train_loss_loader, Label


def test_model(model, loss_dataloader, criterion, device):
    """
    计算所有样本的重构误差
    """
    model.eval()
    model.to(device)
    loss_list = []

    with torch.no_grad():
        for X, X_next in loss_dataloader:
            X = X.to(device)
            X_next = X_next.to(device)

            X_next_hat = model(X)

            # 计算 MSE Loss (Batch, Seq, Feature) -> (Batch)
            loss = torch.mean((X_next - X_next_hat) ** 2, dim=[1, 2])
            loss_list.extend(loss.cpu().numpy())

    return np.array(loss_list)


def calculate_metrics(y_true, y_pred):
    """
    计算 Accuracy, Recall (针对攻击), Specificity (针对正常)
    Label: 0=Attack, 1=Normal
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Recall (查全率): 针对攻击(0) -> tn / (tn+fp)
    # 你的定义：0是攻击。Recall = TP_attack / Total_attack
    if tn + fp == 0:
        recall = 0.0
    else:
        recall = tn / (tn + fp)

    # Specificity (特异性): 针对正常(1) -> tp / (tp+fn)
    if tp + fn == 0:
        specificity = 0.0
    else:
        specificity = tp / (tp + fn)

    # Accuracy
    accuracy = (tn + tp) / (tn + fp + fn + tp + 1e-10)

    return accuracy, recall, specificity


def grid_research(test_subset, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss(reduction='none')

    # 1. 获取 Loss 和 真实标签
    train_loss_loader, Label = prepare_data_loaders(test_subset)
    Label = np.array(Label)  # 0=Attack, 1=Normal

    print("正在计算重构误差分布...")
    loss_ar = test_model(model, train_loss_loader, criterion, device)

    # 2. 确定阈值搜索范围
    min_val = np.min(loss_ar)
    max_val = np.max(loss_ar)
    print(f"Loss Range: [{min_val:.6f}, {max_val:.6f}]")

    # 生成 2000 个对数分布的候选阈值，采样更密集以保证不漏掉最优解
    eps = 1e-10
    thresholds = np.logspace(math.log10(min_val + eps), math.log10(max_val + eps), num=2000)

    # 3. 【核心逻辑修改】直接寻找综合得分最高的阈值
    best_score = -1.0
    best_threshold = 0.0
    # 记录最佳得分对应的三个指标
    best_metrics = {'acc': 0.0, 'rec': 0.0, 'spec': 0.0}

    print("正在进行加权目标寻优 (Weighted Score Optimization)...")
    print("Formula: Score = 0.1*Acc + 0.6*Recall + 0.3*Specificity")

    # 用于记录过程数据以便画图（可选）
    history_t = []
    history_score = []

    for t in thresholds:
        # 预测逻辑：Loss >= Threshold -> 异常(0)，否则 正常(1)
        y_pred = np.where(loss_ar >= t, 0, 1)

        acc, rec, spec = calculate_metrics(Label, y_pred)

        # === 关键修改：直接计算综合得分 ===
        current_score = (0.1 * acc) + (0.6 * rec) + (0.3 * spec)

        history_t.append(t)
        history_score.append(current_score)

        # 如果当前得分比之前的都高，就更新“状元”
        if current_score > best_score:
            best_score = current_score
            best_threshold = t
            best_metrics = {'acc': acc, 'rec': rec, 'spec': spec}

    # 4. 输出最终结果
    final_acc = best_metrics['acc']
    final_rec = best_metrics['rec']
    final_spec = best_metrics['spec']

    print("\n" + "=" * 60)
    print("【最终优化结果】")
    print("-" * 60)
    print(f"Best Weighted Score : {best_score:.4f}")
    print(f"Optimal Threshold   : {best_threshold:.6f}")
    print("-" * 60)
    print(f"Accuracy            : {final_acc:.4f}  (权重 0.1)")
    print(f"Recall (安全/防漏报): {final_rec:.4f}  (权重 0.6)")
    print(f"Specificity (防误报): {final_spec:.4f}  (权重 0.3)")
    print("=" * 60 + "\n")

    # 5. 简单绘制一下得分为何最高的曲线图
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(history_t, history_score, label='Weighted Score', color='blue')
        plt.xscale('log')
        plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best T={best_threshold:.5f}')
        plt.title('Threshold vs. Weighted Score')
        plt.xlabel('Threshold (Log Scale)')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig('score_optimization_curve.png')
        print("优化曲线图已保存为: score_optimization_curve.png")
    except Exception as e:
        pass

    # 返回结果供主程序保存 [Acc, Rec, Spec, Threshold]
    return [final_acc, final_rec, final_spec, best_threshold]