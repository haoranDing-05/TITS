import json
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader

from NLT.NLT_main import likelihood_transformation
from data_processing.timeseries_dataset import loading_survival_dataset, loading_car_hacking, dataClassifier
from model_test.CustomClass import SimpleConcatDataset, SimpleSubset
from model_test.grid_research import prepare_data_loaders, test_model
from model.LSTM import LSTMAutoencoder, train_model
from threshold_utils import SPOTGridSearcher


# =============================================================
# 辅助函数：按时间顺序切分
# =============================================================
def sequential_split(dataset, split_ratio=0.8):
    total_len = len(dataset)
    split_idx = int(total_len * split_ratio)
    indices = list(range(total_len))
    idx1 = indices[:split_idx]
    idx2 = indices[split_idx:]
    return SimpleSubset(dataset, idx1), SimpleSubset(dataset, idx2)


# =============================================================
# 辅助函数：提取 Loss 并做 Log 变换 (统一处理)
# =============================================================
def get_log_scores(dataset, model, criterion, device, epsilon=1e-8):
    """
    计算数据集的 Loss，并进行 Log 变换和微量抖动处理
    返回: log_scores (np.array), labels (np.array)
    """
    loader, _, true_labels = prepare_data_loaders(dataset)
    loss_tensor = test_model(model, loader, criterion, device)
    raw_scores = np.array([l.item() for l in loss_tensor])

    # Log 变换 + 极微量抖动防止重复值
    jitter = np.random.uniform(0, 1e-6, size=raw_scores.shape)
    log_scores = np.log(raw_scores + epsilon + jitter)

    # 提取标签 (确保是 numpy array)
    # 【修复重点】：原始数据中 0 代表 Attack，1 代表 Normal。
    # SPOT 需要 1=Anomaly (Attack), 0=Normal。
    # 所以必须执行反转：(labels == 0) -> 1
    raw_labels = np.array(true_labels)
    spot_labels = (raw_labels == 0).astype(int)

    return log_scores, spot_labels


# =============================================================
# 新评估函数：Method B (Val 调参 -> Test 验证)
# =============================================================
def evaluate_dataset_method_b(dataset_name, val_dataset, test_dataset, calib_dataset, model, device, criterion):
    print(f"\n" + "=" * 80)
    print(f"执行 Method B 评估: {dataset_name}")
    print(f"流程: Calibration(纯正常) -> Validation(调优 q) -> Test(一次性测试)")
    print("=" * 80)

    # 1. 准备 Calibration 数据 (用于 SPOT 初始化)
    # ---------------------------------------------------------
    print(f"[{dataset_name}] 正在处理 Calibration 数据...")
    # 注意：get_log_scores 内部会处理标签，虽然 Calib 只有 Normal，但这里我们只取 scores
    calib_scores, _ = get_log_scores(calib_dataset, model, criterion, device)

    # 截取一部分作为初始化，防止过大 (例如取前 5000 个点)
    init_size = 5000
    init_data_log = calib_scores[:init_size] if len(calib_scores) > init_size else calib_scores
    print(f"  Calibration 数据量: {len(init_data_log)}")

    # 2. 在 Validation 集上寻找最佳 q 值
    # ---------------------------------------------------------
    print(f"\n[{dataset_name}] 正在 Validation 集上搜索最佳参数...")
    val_scores, val_labels = get_log_scores(val_dataset, model, criterion, device)

    # 初始化搜索器，init_ratio=0 表示全部 val_scores 都用于搜索，不再次切分
    # 关键：手动注入 init_data
    spot_searcher_val = SPOTGridSearcher(val_scores, val_labels, init_ratio=0)
    spot_searcher_val.init_data = init_data_log

    # 定义搜索网格
    q_grid = [1e-2, 5e-3, 3e-3, 2e-3, 1e-3, 5e-4, 1e-4, 1e-5]
    level = 0.98  # 固定 level

    best_params, best_val_metrics = spot_searcher_val.search(q_grid, [level], save_all_results=False)

    best_q = best_params['q']
    print(f"\n>>> [Validation 结果] 最佳 q={best_q}, F1={best_val_metrics['f1']:.4f}")

    # 3. 在 Test 集上使用最佳 q 值进行最终评估
    # ---------------------------------------------------------
    print(f"\n[{dataset_name}] 正在 Test 集上使用 q={best_q} 进行最终测试 (无泄露)...")
    test_scores, test_labels = get_log_scores(test_dataset, model, criterion, device)

    spot_searcher_test = SPOTGridSearcher(test_scores, test_labels, init_ratio=0)
    spot_searcher_test.init_data = init_data_log  # 使用同样的初始化数据

    # 只跑一次 best_q
    final_params, final_metrics = spot_searcher_test.search([best_q], [level], save_all_results=False)

    # 还原阈值 (Log -> Linear) 用于展示
    EPSILON = 1e-8
    log_th = final_metrics['threshold']
    real_th = np.exp(log_th) - EPSILON

    print(f"-" * 60)
    print(f"{'Metric':<15} | {'Value'}")
    print(f"-" * 60)
    print(f"{'Final F1':<15} | {final_metrics['f1']:.4f}")
    print(f"{'Precision':<15} | {final_metrics['precision']:.4f}")
    print(f"{'Recall':<15} | {final_metrics['recall']:.4f}")
    print(f"{'Accuracy':<15} | {final_metrics['accuracy']:.4f}")
    print(f"{'Threshold':<15} | {real_th:.6f}")
    print(f"-" * 60)

    # 保存结果
    res_entry = {
        "dataset": dataset_name,
        "method": "B_Validation_Tuning",
        "best_q_from_val": best_q,
        "test_metrics": final_metrics
    }
    filename = f'result_method_b_{dataset_name.split()[0].lower()}.json'
    with open(filename, 'w') as f:
        json.dump(res_entry, f, indent=4)
    print(f"结果已保存至: {filename}")


if __name__ == '__main__':
    # 基础配置
    batch_size = 100
    input_size = 1
    hidden_size = 50
    epoch = 500
    window_size = 10
    sliding_window = 0.05
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transfer = likelihood_transformation()

    # =========================================================
    # Phase 1: Training (Survival)
    # =========================================================
    print(">>> Phase 1: Training (Survival)...")
    surv_data = loading_survival_dataset(window_size, sliding_window, transfer, 'all')
    surv_norm, surv_attk = dataClassifier(surv_data)

    # 80% 用于训练模型
    train_norm, test_norm_surv = sequential_split(surv_norm, 0.8)

    model = LSTMAutoencoder(input_size, hidden_size, device)
    train_loader = DataLoader(train_norm, batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # 如果已有模型想跳过训练，可以注释掉下面这行并加载权重
    model = train_model(model, train_loader, criterion, optimizer, epoch, device)

    # =========================================================
    # Phase 3: Car-Hacking 测试 (Method B: Train/Val/Test Split)
    # =========================================================
    print("\n>>> Phase 3: Car-Hacking (Method B)...")
    ch_data = loading_car_hacking(window_size, sliding_window, transfer, 'all')
    ch_norm, ch_attk = dataClassifier(ch_data)

    # --- 数据划分逻辑 ---
    # 1. 从正常数据中切出 Calibration Set (例如前 80%)
    ch_calib, ch_norm_remain = sequential_split(ch_norm, 0.7)

    # 2. 将剩余的正常数据切分为 Val (50%) 和 Test (50%)
    ch_val_norm, ch_test_norm = sequential_split(ch_norm_remain, 0.6)

    # 3. 将攻击数据也切分为 Val (50%) 和 Test (50%)
    ch_val_attk, ch_test_attk = sequential_split(ch_attk, 0.5)

    # 4. 组装最终数据集
    val_dataset = SimpleConcatDataset([ch_val_norm, ch_val_attk])
    test_dataset = SimpleConcatDataset([ch_test_norm, ch_test_attk])

    print(f"数据统计:")
    print(f"  Calib (Normal): {len(ch_calib)}")
    print(f"  Val   (Mixed) : {len(val_dataset)} (Norm:{len(ch_val_norm)}, Attk:{len(ch_val_attk)})")
    print(f"  Test  (Mixed) : {len(test_dataset)} (Norm:{len(ch_test_norm)}, Attk:{len(ch_test_attk)})")

    # 执行 Method B 评估
    evaluate_dataset_method_b("Car-Hacking", val_dataset, test_dataset, ch_calib, model, device, criterion)