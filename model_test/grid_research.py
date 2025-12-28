import torch
import numpy as np
import math
from torch import nn
from model.LSTM import LSTMAutoencoder
from torch.utils.data import DataLoader
from model_test.CustomClass import SimpleSubset, CustomDataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def AUC(y_true, y_pred):  #计算AUC指标 输入真指标与预测指标两个列表 指标的集合含义是距离左上角的距离
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if tp + fn == 0 or fp + tn == 0:
        return 0
    else:
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        return np.sqrt((1 - tpr) ** 2 + fpr ** 2)


def myf1_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if tp + fn == 0 or fp + tn == 0:
        return 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return 2 * precision * recall / (precision + recall), [precision, recall, f1, accuracy]


def myaccuracy_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if tp + fn == 0 or fp + tn == 0:
        return 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return (tp + tn) / (tp + tn + fp + fn), [precision, recall, f1, accuracy]


def g_mean(y_true, y_pred):  #计算G_Mean指标 输入真指标与预测指标两个列表
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if tp + fn == 0 or fp + tn == 0:
        return 0
    else:
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return np.sqrt(tpr * (1 - fpr)), [precision, recall, f1, accuracy]


def prepare_data_loaders(train_dataset):  #返回（X,Label)的数据集
    train_loss_loader = DataLoader(train_dataset)
    indice = [i for i in range(len(train_dataset))]
    train_dataset = SimpleSubset(train_dataset, indice)
    X = []
    Label = []
    for i in range(len(train_dataset)):
        x, label = train_dataset.get_test_sample(i)
        X.append(x)
        Label.append(label)
    train_label_loader = CustomDataset(X, Label)
    return train_loss_loader, train_label_loader, Label


def test_model(model, loss_dataloader, criterion, device):  #遍历训练集生成重建偏差的列表
    model.to(device)
    loss_list = []
    for X, X_next in loss_dataloader:
        X = X.to(device)
        X_next = X_next.to(device)
        X_next_hat = model(X)
        loss = criterion(X_next, X_next_hat)
        loss_list.append(loss)
    return loss_list


def generate_predictions(Loss, params_grid):
    #Label_hat_np的形状是（阈值个数，每个阈值判断条件下的判断结果的个数）
    Label_hat = []
    for threshold in params_grid['threshold']:
        row = []
        for loss in Loss:
            if loss >= threshold:
                label_hat = 0
            else:
                label_hat = 1
            row.append(label_hat)
        Label_hat.append(row)
    Label_hat_np = np.array(Label_hat)
    #print(Label_hat_np.shape)#（91，734）
    return Label_hat_np


def tolist(array):
    my_list = []
    for element in array:
        my_list.append(element)
    return my_list


def evaluate_metrics(Label, Label_hat, params_grid, result):
    #不同阈值中寻找，不同指标分别达到最大时阈值的值
    best_accuracy = 0
    best_accuracy_threshold = 0
    best_f1 = 0
    best_f1_threshold = 0
    best_gmean = 0
    best_gmean_threshold = 0

    Label = tolist(Label)

    for i, label_hat in enumerate(Label_hat):

        label_hat = tolist(label_hat)
        accuracy, best_acc_ar = myaccuracy_score(Label, label_hat)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_accuracy_threshold = params_grid['threshold'][i]
            result["accuracy"] = best_acc_ar

        f1, best_f1_ar = myf1_score(Label, label_hat)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = params_grid['threshold'][i]
            result["f1"] = best_f1_ar

        gmean, best_gmean_ar = g_mean(Label, label_hat)
        if gmean > best_gmean:
            best_gmean = gmean
            best_gmean_threshold = params_grid['threshold'][i]
            result["gmean"] = best_gmean_ar

    result["accuracy"].append(best_accuracy)
    result["accuracy"].append(best_accuracy_threshold)
    result["f1"].append(best_f1)
    result["f1"].append(best_f1_threshold)
    result["gmean"].append(best_gmean)
    result["gmean"].append(best_gmean_threshold)


def calculate_AUC_distance(Label, Label_hat, params_grid):  #不同阈值中寻找，AUC指标达到最大时阈值的值
    AUC_distance = []
    for i in range(len(params_grid['threshold'])):
        AUC_distance.append(AUC(Label, Label_hat[i]))
    AUC_min = min(AUC_distance)
    AUC_index = np.argmin(AUC_distance)
    print(f"auc最小值是 {AUC_min}，对应的索引是 {AUC_index}")
    print(f"auc对应最佳阈值:{params_grid['threshold'][AUC_index]}")


def grid_research(test_subset, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()

    # 准备数据加载器
    #这里都没有打乱顺序，所以一个索引对应的数据是关联的
    train_loss_loader, train_label_loader, Label = prepare_data_loaders(test_subset)

    # 测试模型并获取损失列表
    Loss = test_model(model, train_loss_loader, criterion, device)
    loss_ar = show_Loss(Loss, Label)

    max_value = np.amax(loss_ar)
    min_value = np.amin(loss_ar)

    #阈值预设为一个指数增长的序列，如果线性增长
    max_log = math.log(max_value)
    min_log = math.log(min_value)

    #候选的阈值从loss的最小值到最大值 90等分
    threshold_ar = np.arange(min_log, max_log, (max_log - min_log) / 90)
    for i in range(len(threshold_ar)):
        threshold_ar[i] = math.exp(threshold_ar[i])
    params_grid = {'threshold': threshold_ar}
    # 生成预测结果
    Label_hat_np = generate_predictions(Loss, params_grid)

    result = {"accuracy": [], "f1": [], "gmean": []}
    # 评估指标
    evaluate_metrics(Label, Label_hat_np, params_grid, result)

    # 计算 AUC 距离
    calculate_AUC_distance(Label, Label_hat_np, params_grid)

    print("准确率最大值: {:.4f}, 对应的阈值: {:.4f},precision:{:.4f},recal:{:.4f},f1:{:.4f},acc:{:.4f}".format(
        result["accuracy"][4], result["accuracy"][5], result["accuracy"][0], result["accuracy"][1],
        result["accuracy"][2], result["accuracy"][3]))
    print("f1最大值: {:.4f}, 对应的阈值: {:.4f},precision:{:.4f},recal:{:.4f},f1:{:.4f},acc:{:.4f}".format(
        result["f1"][4], result["f1"][5], result["f1"][0], result["f1"][1], result["f1"][2], result["f1"][3]))
    print("gmean最大值: {:.4f}, 对应的阈值: {:.4f},precision:{:.4f},recal:{:.4f},f1:{:.4f},acc:{:.4f}".format(
        result["gmean"][4], result["gmean"][5], result["gmean"][0], result["gmean"][1], result["gmean"][2],
        result["gmean"][3]))

    return [result["accuracy"][4], result["f1"][4], result["gmean"][4], result["f1"][5]]


def show_Loss(loss, labels):
    length = len(loss)
    x = np.arange(length)
    loss_ar = np.array([tensor.detach().cpu().numpy() for tensor in loss])
    sample_color = []
    widths = []
    for label in labels:
        if label == 0:
            sample_color.append("red")
            widths.append(1)
        else:
            sample_color.append("green")
            widths.append(1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, loss_ar, color=sample_color, width=widths)
    ax.set_yscale('log')
    plt.show()
    return loss_ar
