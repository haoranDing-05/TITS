import numpy as np
from scipy.stats import genpareto
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import itertools
from tqdm import tqdm
import sys

class SPOTDetector:
    """
    SPOT (Streaming Peaks Over Threshold) 动态阈值检测器 - 调试增强版
    """
    def __init__(self, q=1e-3, init_level=0.98):
        self.q = q
        self.init_level = init_level
        self.init_threshold = None 
        self.peaks = []            
        self.n = 0                 
        self.Nt = 0                
        self.sigma = 0
        self.gamma = 0
        self.current_z = 0         

    def fit_initial(self, init_data):
        data = np.array(init_data)
        self.n = len(data)
        self.init_threshold = np.percentile(data, 100 * self.init_level)
        peaks = data[data > self.init_threshold] - self.init_threshold
        self.peaks = peaks.tolist()
        self.Nt = len(self.peaks)
        print(f"\n[Init] 初始化完成. 初始阈值: {self.init_threshold:.6f}, 初始Peaks数量: {self.Nt}")
        self._update_gpd_params()
        self._update_threshold()

    def step(self, x):
        self.n += 1
        is_anomaly = False
        if x > self.current_z:
            is_anomaly = True
        
        # 只有超过初始阈值的数据才用于更新分布
        if x > self.init_threshold:
            self.peaks.append(x - self.init_threshold)
            self.Nt += 1
            
            # === 策略：批量更新以防止卡死，同时输出调试信息 ===
            # 如果是 Survival 数据集（数据少），或者处于初期阶段，频繁更新
            # 如果是 Car-Hacking 数据集（数据多），每 50-100 次更新一次
            should_update = (self.Nt < 100) or (self.Nt % 100 == 0)
            
            if should_update:
                # 打印当前进度，证明程序活着
                # sys.stdout.write(f"\r[Running] 处理第 {self.n} 个数据点 | 积累 Peaks: {self.Nt} -> 正在拟合...")
                # sys.stdout.flush()
                
                self._update_gpd_params()
                self._update_threshold()
            
        return is_anomaly, self.current_z

    def _update_gpd_params(self):
        # 只有 Peaks 数量足够才进行拟合
        if self.Nt >= 10:
            try:
                # 尝试拟合 GPD 分布
                params = genpareto.fit(self.peaks, floc=0)
                self.gamma = params[0]
                self.sigma = params[2]
                
                # === 【调试输出：成功】 ===
                # 注意：为了避免刷屏太快，你可以选择性注释掉下面这行
                # print(f" ✅ [Fit Success] Nt={self.Nt} | gamma={self.gamma:.4f} | sigma={self.sigma:.4f}")
                
            except Exception as e:
                # === 【调试输出：失败】 ===
                # 捕获并打印具体的错误原因
                self.gamma = 0  # 重置为保守值
                self.sigma = np.std(self.peaks) if len(self.peaks) > 0 else 1.0
                print(f" ❌ [Fit Failed] Nt={self.Nt} | Error: {e}")
        else:
            # Peaks 太少，跳过
            # print(f" ⚠️ [Fit Skipped] Nt={self.Nt} (Need >= 10)")
            self.gamma = 0.1
            self.sigma = np.std(self.peaks) if len(self.peaks) > 0 else 1.0

    def _update_threshold(self):
        try:
            if self.Nt == 0:
                self.current_z = self.init_threshold * (1 + self.q * 100)
            elif self.gamma == 0 or abs(self.gamma) < 1e-10:
                self.current_z = self.init_threshold * (1 + self.q * 100)
            else:
                r1 = (self.q * self.n) / max(self.Nt, 1)
                if r1 <= 0 or r1 >= 1:
                    self.current_z = self.init_threshold * (1 + self.q * 100)
                else:
                    r2 = pow(r1, -self.gamma) - 1
                    if np.isnan(r2) or np.isinf(r2) or r2 < 0:
                        self.current_z = self.init_threshold * (1 + self.q * 100)
                    else:
                        threshold_adjustment = (self.sigma / self.gamma) * r2
                        self.current_z = self.init_threshold + threshold_adjustment
                        if self.current_z < self.init_threshold:
                            self.current_z = self.init_threshold * (1 + self.q * 100)
                            
        except Exception as e:
            print(f" ❌ [Threshold Calc Error] {e}")
            self.current_z = self.init_threshold * (1 + self.q * 100)

class SPOTGridSearcher:
    # ... (保持原来的 __init__ 不变) ...
    def __init__(self, scores, labels, init_ratio=0.3):
        self.scores = np.array(scores)
        self.labels = np.array(labels)
        split_idx = int(len(scores) * init_ratio)
        self.init_data = self.scores[:split_idx]
        self.eval_data = self.scores[split_idx:]
        self.eval_labels = self.labels[split_idx:]

    def run_simulation(self, q, level):
        print(f"\n--- 开始模拟: q={q}, level={level} ---")
        detector = SPOTDetector(q=q, init_level=level)
        detector.fit_initial(self.init_data)
        preds = []
        final_thresholds = [] 
        
        # 使用 tqdm 显示总体进度条
        for x in tqdm(self.eval_data, desc="Stream Processing", mininterval=1.0):
            is_anom, current_z = detector.step(x)
            preds.append(int(is_anom))
            final_thresholds.append(current_z)
            
        return np.array(preds), final_thresholds[-1] if final_thresholds else detector.current_z

    def search(self, q_grid=None, level_grid=None, save_all_results=False):
        # ... (和之前一样，但记得加上我之前提到的 threshold 修复) ...
        if q_grid is None: q_grid = [1e-3, 1e-4]
        if level_grid is None: level_grid = [0.98, 0.99]

        best_f1 = -1
        best_params = {}
        best_metrics = {}
        all_results_by_q = {}
        
        total_combs = list(itertools.product(q_grid, level_grid))
        
        for q, level in total_combs: # 移除了外层的 tqdm，因为 run_simulation 里面已经有 tqdm 了，防止嵌套进度条乱掉
            preds, final_threshold = self.run_simulation(q, level)
            current_f1 = f1_score(self.eval_labels, preds)
            current_precision = precision_score(self.eval_labels, preds)
            current_recall = recall_score(self.eval_labels, preds)
            current_accuracy = accuracy_score(self.eval_labels, preds)
            
            # 打印每次搜索的摘要
            print(f"Result for q={q}: F1={current_f1:.4f}, Th={final_threshold:.6f}")

            result_entry = {
                'q': q,
                'level': level,
                'precision': float(current_precision),
                'recall': float(current_recall),
                'f1': float(current_f1),
                'accuracy': float(current_accuracy),
                'final_threshold': float(final_threshold)
            }
            
            q_key = str(q)
            if q_key not in all_results_by_q:
                all_results_by_q[q_key] = []
            all_results_by_q[q_key].append(result_entry)
            
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_params = {'q': q, 'level': level}
                best_metrics = {
                    'precision': current_precision,
                    'recall': current_recall,
                    'f1': current_f1,
                    'accuracy': current_accuracy,
                    'threshold': final_threshold # <--- 记得这里修复了之前说的 Bug
                }
        
        if save_all_results:
            return best_params, best_metrics, all_results_by_q
        else:
            return best_params, best_metrics