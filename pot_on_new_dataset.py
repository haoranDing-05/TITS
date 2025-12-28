import json
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from NLT.NLT_main import likelihood_transformation
from data_processing.timeseries_dataset import loading_survival_dataset, loading_car_hacking, dataClassifier
from model_test.CustomClass import SimpleConcatDataset, SimpleSubset

from model_test.grid_research import grid_research, prepare_data_loaders, test_model
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
# 封装评估函数 (Scheme C: Log-Transform)
# =============================================================
def evaluate_dataset_scheme_c(dataset_name, test_dataset, model, device, criterion, pure_normal_dataset=None):
    print(f"\n" + "="*60)
    print(f"正在评估数据集: {dataset_name} (Scheme C: Log-Transform)")
    print("="*60)

    # --- Step 1: 静态基准 ---
    static_results = grid_research(test_dataset, model)
    if len(static_results) >= 4:
        best_static_f1 = static_results[1]
        best_static_threshold = static_results[3]
    else:
        best_static_f1 = static_results[1]
        best_static_threshold = 0.0017 
    print(f">>> [Static] F1={best_static_f1:.4f}, Th={best_static_threshold:.6f}")

    # --- Step 2: 准备数据 (Log 变换) ---
    # 定义一个极小值防止 log(0)
    EPSILON = 1e-8
    print(f"[{dataset_name}] Step 2: 执行对数变换 log(x + {EPSILON})...")
    
    # 获取测试集 Loss
    test_loss_loader, _, true_labels_list = prepare_data_loaders(test_dataset)
    loss_list_tensor = test_model(model, test_loss_loader, criterion, device)
    raw_test_scores = np.array([l.item() for l in loss_list_tensor])
    
    # 【核心修改】对数变换 + 微量抖动
    # 对数能把极其偏斜的分布拉正，极大概率解决拟合失败问题
    jitter = np.random.uniform(0, 1e-6, size=raw_test_scores.shape)
    test_scores_log = np.log(raw_test_scores + EPSILON + jitter)
    
    spot_labels = (np.array(true_labels_list) == 0).astype(int) 

    # 准备校准集
    init_data_log = None
    if pure_normal_dataset:
        normal_loader = DataLoader(pure_normal_dataset, batch_size=1024)
        norm_loss_tensor = test_model(model, normal_loader, criterion, device)
        raw_calib_scores = np.array([l.item() for l in norm_loss_tensor])
        
        # 同样做对数变换
        init_size = 5000 # 尽可能多取点
        if len(raw_calib_scores) > init_size:
            raw_init = raw_calib_scores[:init_size]
        else:
            raw_init = raw_calib_scores
            
        jitter_init = np.random.uniform(0, 1e-6, size=raw_init.shape)
        init_data_log = np.log(raw_init + EPSILON + jitter_init)
    else:
        init_data_log = test_scores_log[:2000]

    # --- Step 3: SPOT 遍历测试 ---
    # Log 空间下，数据分布更平滑，我们可以尝试用回 0.98，或者保守点用 0.90
    final_level = 0.90 
    print(f"[{dataset_name}] Step 3: SPOT 遍历测试 (Level={final_level}, Log-Space)...")
    
    spot_searcher = SPOTGridSearcher(test_scores_log, spot_labels)
    spot_searcher.init_data = init_data_log
    
    q_grid = [1e-2, 5e-3, 3e-3, 2e-3, 1e-3, 5e-4, 1e-4, 1e-5]
    
    detailed_results = []
    best_spot_metrics = None

    print(f"{'q-Value':<10} | {'F1-Score':<10} | {'Recall':<10} | {'Real Th'}")
    print("-" * 60)

    for q in q_grid:
        try:
            params, metrics = spot_searcher.search([q], [final_level])
            
            # 【还原阈值】从 Log 空间变回线性空间
            # Th_real = exp(Th_log) - epsilon
            log_threshold = metrics.get('threshold', 0.0)
            
            # 如果拟合失败，log_threshold 会是初始分位点(log space)，还原回去也是对的
            real_threshold = np.exp(log_threshold) - EPSILON
            
            th_display = f"{real_threshold:.6f}"

            res_entry = {
                "q": q,
                "level": final_level,
                "f1": metrics['f1'],
                "precision": metrics['precision'],
                "recall": metrics['recall'],
                "threshold": real_threshold,
                "log_threshold": log_threshold
            }
            detailed_results.append(res_entry)
            
            print(f"{q:<10} | {metrics['f1']:.4f}     | {metrics['recall']:.4f}     | {th_display}")

            if best_spot_metrics is None or metrics['f1'] > best_spot_metrics['f1']:
                best_spot_metrics = metrics
                best_spot_metrics['threshold'] = real_threshold 

        except Exception as e:
            print(f"Error testing q={q}: {e}")

    # 保存
    detail_filename = f'spot_details_C_{dataset_name.split()[0].lower()}.json'
    with open(detail_filename, 'w') as f:
        json.dump({"dataset": dataset_name, "results": detailed_results}, f, indent=4)
    
    return {'spot_metrics': best_spot_metrics}

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

    # 1. Survival 加载 & 训练
    print(">>> Phase 1: Training...")
    surv_data = loading_survival_dataset(window_size, sliding_window, transfer, 'all')
    surv_norm, surv_attk = dataClassifier(surv_data)
    train_norm, test_norm_surv = sequential_split(surv_norm, 0.8)
    
    model = LSTMAutoencoder(input_size, hidden_size, device)
    train_loader = DataLoader(train_norm, batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    model = train_model(model, train_loader, criterion, optimizer, epoch, device)

    # 2. Survival 测试 (Scheme C)
    survival_test = SimpleConcatDataset([test_norm_surv, surv_attk])
    evaluate_dataset_scheme_c("Survival", survival_test, model, device, criterion, test_norm_surv)

    # 3. Car-Hacking 测试 (Scheme C)
    print("\n>>> Phase 3: Car-Hacking...")
    ch_data = loading_car_hacking(window_size, sliding_window, transfer, 'all')
    ch_norm, ch_attk = dataClassifier(ch_data)
    ch_calib, ch_eval = sequential_split(ch_norm, 0.2)
    ch_test = SimpleConcatDataset([ch_eval, ch_attk])
    
    evaluate_dataset_scheme_c("Car-Hacking", ch_test, model, device, criterion, ch_calib)