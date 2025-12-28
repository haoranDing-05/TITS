from collections import deque, defaultdict

import numpy as np


def calculate_combined_stats(sizes, means, stds):
    # 验证输入的合法性
    if not (len(means) == len(stds) == len(sizes)):
        raise ValueError("均值、标准差和样本量的列表长度必须相同")

    # 计算总体均值
    total_size = sum(sizes)
    combined_mean = sum(m * n for m, n in zip(means, sizes)) / total_size

    # 计算总体标准差
    sum_sq_dev = sum(n * (std ** 2 + (m - combined_mean) ** 2) for m, std, n in zip(means, stds, sizes))
    combined_std = np.sqrt(sum_sq_dev / total_size)

    return combined_mean, combined_std


class StrideNode:
    def __init__(self, stride):
        self.stride = stride
        self.unique_id = set()
        self.temp_dlc = defaultdict(int)
        self.number = 0
        self.label = 1
        self.start_time = None

    def add_data(self, new_data):
        if self.start_time is None:
            self.start_time = new_data[0]
        if new_data[-1] == 'T':
            self.label = 0
        self.unique_id.add(new_data[1])
        self.temp_dlc[new_data[3]] += 1
        self.number += 1

    def get_result(self):
        return [self.label, self.unique_id, self.temp_dlc, self.number]

    def is_full(self, new_data):
        if self.number == 0:
            return False
        if new_data[0] - self.start_time > self.stride:
            return True
        else:
            return False


#
#
#
# -----------------------------------------------------------------------------
# [旧版 CarQueue - 已弃用]
# 原始实现：每次 get_result() 时全量遍历队列，计算 10 个切片窗口。
# 复杂度 O(N)，对于高频 stride (0.1s) 性能开销较大。
# -----------------------------------------------------------------------------
# class CarQueue(deque):
#    def __init__(self, max_len, stride, window_size=3):
#        self.max_len = max_len
#        self.node_len = int(window_size / stride)
#        self.before = None
#        super().__init__(maxlen=max_len)
#
#    def get_result(self):
#        index = 0
#        results = []
#        labels = []
#        while index < self.max_len:
#            number = 0
#            label = 1
#            unique_id = set()
#            temp_dlc = defaultdict(int)
#            for i in range(self.node_len):
#                # print(temp)
#                temp = self[index].get_result()
#                for can in temp[1]:
#                    unique_id.add(can)
#                for k, v in temp[2].items():
#                    temp_dlc[k] += v
#                if temp[0] == 0:
#                    label = 0
#                index += 1
#                number += temp[3]
#            temp_dlc_1 = np.array([v for v in temp_dlc.values()])
#            if number > 50:
#                temp_res = [len(unique_id), np.mean(temp_dlc_1), np.std(temp_dlc_1)]
#                self.before = temp_res
#            else:
#                temp_res = self.before
#            results.append(temp_res)
#            labels.append(label)
#
#            # print(results)
#        print(len(results))
#        return results, labels

# -----------------------------------------------------------------------------
# [新版 CarQueue - 优化版]
# 优化思路：
# 1. 维护 10 个并行的滑动窗口统计状态 (num_windows)。
# 2. 利用 deque 的移位特性，在 append() 前进行增量更新 (O(1))。
# 3. 避免重复遍历，直接输出维护好的统计结果。
# -----------------------------------------------------------------------------
class CarQueue(deque):
    def __init__(self, max_len, stride, window_size=3):
        """
        :param max_len: 队列最大长度，例如 300 (对应 30s 数据，stride=0.1s)
        :param stride: 单个 StrideNode 的时间跨度，例如 0.1s
        :param window_size: 单个特征窗口的时间长度，例如 3s
        """
        super().__init__(maxlen=max_len)
        self.max_len = max_len
        self.stride = stride
        self.window_size = window_size

        # node_len: 一个 3s 窗口包含多少个 stride 节点 (3 / 0.1 = 30 个)
        self.node_len = int(window_size / stride)

        # num_windows: 整个队列 (30s) 能切分成多少个 3s 窗口 (300 / 30 = 10 个)
        # 这 10 个窗口在逻辑上是“首尾相接”的，覆盖整个队列长度
        self.num_windows = max_len // self.node_len

        # 维护 10 个窗口的统计状态列表
        # 每个窗口维护自己的：样本数(n), DLC出现次数字典(dlc_counts), 唯一ID计数(ids), 攻击标签计数(attack_cnt)
        # 注意：dlc_counts 是 {DLC值: 出现次数}，用于计算出现次数的均值和标准差（与旧版逻辑一致）
        self.window_stats = []
        for _ in range(self.num_windows):
            self.window_stats.append({
                'n': 0,
                'dlc_counts': defaultdict(int),  # {DLC值: 出现次数}
                'ids': defaultdict(int),  # 唯一ID计数
                'attack_cnt': 0  # 有多少个内部的stride_node是攻击标签
            })

        # 用于数据不足时的特征填充
        self.before = None

    def append(self, new_node):
        """
        重写 append 方法，实现 O(1) 的增量更新。
        逻辑：
        在 deque 真正发生物理移位（把最左边的挤出去，最右边的加进来）之前，
        我们先计算这次移位对那 10 个窗口统计量的影响。
        """

        # 只有当队列已满时，才进入稳定的“流水线”模式
        if len(self) == self.max_len:
            # --- 增量更新流水线 ---
            # 遍历所有 10 个窗口，每个窗口都要“吐出旧的，吃进新的”
            for k in range(self.num_windows):
                # 1. 确定“即将滑出”该窗口的节点 (node_out)
                #    由于 deque 还没变动，self[k * 30] 就是该窗口当前最左边的节点
                #    也就是 update 后不再属于该窗口的节点
                idx_out = k * self.node_len
                node_out = self[idx_out]

                # 2. 确定“即将滑入”该窗口的节点 (node_in)
                #    如果是最后一个窗口 (k=9)，新节点 new_node 将滑入
                #    如果是其他窗口 (k=0到8)，其右侧相邻窗口的第一个节点将滑入 (self[(k+1)*30])
                if k == self.num_windows - 1:
                    node_in = new_node
                else:
                    idx_in = (k + 1) * self.node_len
                    node_in = self[idx_in]

                # 3. 更新第 k 个窗口的统计量
                self._update_window(self.window_stats[k], node_out, node_in)

        # 执行实际的 deque 入队操作 (最左元素自动被移除)
        super().append(new_node)

        # 特殊情况处理：队列刚满的那一瞬间，统计量还没初始化
        # (因为上面的 if len == max_len 还没进去过)
        if len(self) == self.max_len and self.window_stats[0]['n'] == 0:
            self._init_all_stats()

    def _update_window(self, stats, node_out, node_in):
        """
        stats: 一个窗口的统计量
        node_out: 即将滑出的节点
        node_in: 即将滑入的节点
        更新单个窗口的统计量：减去 node_out，加上 node_in
        """
        # --- 减去 node_out ---
        res_out = node_out.get_result()  # [label, ids, dlc_dict, num]

        # 减去node_out的样本数
        stats['n'] -= res_out[3]
        # 维护有多少个内部的stride_node是攻击标签
        if res_out[0] == 0:
            stats['attack_cnt'] -= 1

        for uid in res_out[1]:
            stats['ids'][uid] -= 1
            if stats['ids'][uid] == 0:
                del stats['ids'][uid]  # 保持字典精简

        # 更新 DLC 出现次数字典（减去旧的）
        # res_out[2] 是 {DLC值: 出现次数}
        for dlc_val, count in res_out[2].items():
            stats['dlc_counts'][dlc_val] -= count
            if stats['dlc_counts'][dlc_val] == 0:
                del stats['dlc_counts'][dlc_val]  # 保持字典精简

        # --- 加上 node_in ---
        res_in = node_in.get_result()

        stats['n'] += res_in[3]
        if res_in[0] == 0:
            stats['attack_cnt'] += 1
        for uid in res_in[1]:
            stats['ids'][uid] += 1

        # 更新 DLC 出现次数字典（加上新的）
        for dlc_val, count in res_in[2].items():
            stats['dlc_counts'][dlc_val] += count

    def _init_all_stats(self):
        """
        全量初始化：当队列刚填满时，遍历所有数据计算初始状态
        """
        for k in range(self.num_windows):
            start_idx = k * self.node_len
            end_idx = (k + 1) * self.node_len

            stats = self.window_stats[k]
            # 重置
            stats['n'] = 0
            stats['dlc_counts'] = defaultdict(int)
            stats['ids'] = defaultdict(int)
            stats['attack_cnt'] = 0

            # 累加该窗口范围内的所有节点
            for i in range(start_idx, end_idx):
                node = self[i]
                res = node.get_result()

                stats['n'] += res[3]
                if res[0] == 0: stats['attack_cnt'] += 1
                for uid in res[1]:
                    stats['ids'][uid] += 1
                # 累加 DLC 出现次数（与旧版逻辑一致）
                for dlc_val, count in res[2].items():
                    stats['dlc_counts'][dlc_val] += count

    def get_result(self):
        """
        输出当前时刻 10 个窗口的特征
        与旧版逻辑一致：计算 DLC 出现次数的均值和标准差
        """
        results = []
        labels = []

        for k in range(self.num_windows):
            stats = self.window_stats[k]

            # 只有样本数足够才计算统计特征
            if stats['n'] > 50:
                # 与旧版逻辑一致：temp_dlc_1 = np.array([v for v in temp_dlc.values()])
                # 即：把所有 DLC 出现次数取出来，计算均值和标准差
                dlc_counts_array = np.array(list(stats['dlc_counts'].values()))

                if len(dlc_counts_array) > 0:
                    mean = np.mean(dlc_counts_array)
                    std = np.std(dlc_counts_array)
                else:
                    mean = 0.0
                    std = 0.0

                unique_cnt = len(stats['ids'])

                temp_res = [unique_cnt, mean, std]
                self.before = temp_res  # 简单缓存最后一个有效值
            else:
                # 样本不足，使用上一次的有效值填充
                temp_res = self.before if self.before else [0, 0, 0]

            # 只要窗口内有任意攻击节点，标签即为 0 (攻击)
            label = 0 if stats['attack_cnt'] > 0 else 1

            results.append(temp_res)
            labels.append(label)

        # 调试打印，可保留或注释
        # print(len(results))
        return results, labels
