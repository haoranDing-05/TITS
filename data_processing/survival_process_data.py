import time
from collections import deque
import sys
import os

import numpy as np
import pandas as pd

# 添加项目根目录到路径，以便导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

#from data_processing.car_queue import TimeSlidingWindow
from data_processing.new2_car_queue import StrideNode, CarQueue


def detect_survival_file_format(file_path):
    """
    检测Survival数据集文件格式
    返回: ('csv_like' 或 'mixed', is_attack)
    csv_like: 逗号分隔格式（Flooding/Fuzzy/Malfunction）
    mixed: 混合格式，前3列逗号分隔，数据部分空格分隔（FreeDriving）
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if not first_line:
                return None, None
            
            # 判断是否为FreeDriving（正常数据）
            if 'FreeDriving' in file_path or 'freedriving' in file_path.lower():
                return 'mixed', 0  # FreeDriving是正常数据

            # 判断是否为CSV格式（逗号分隔所有字段）
            # CSV格式：所有字段都用逗号分隔，包括数据字节部分
            # 例如：1513921815.350702,0081,8,7F,84,69,00,00,00,00,69,R
            if ',' in first_line:
                return 'csv_like', 1  # Flooding/Fuzzy/Malfunction是攻击数据

            return None, None
    except Exception as e:
        print(f"检测文件格式时出错: {e}")
        return None, None


def parse_csv_like_format(file_path):
    """
    解析CSV格式的Survival数据（Flooding/Fuzzy/Malfunction）
    格式：时间戳,ID,DLC,数据字节1,数据字节2,...,方向标识
    注意：由于DLC值不同，数据字节数可能不同，导致列数不一致
    """
    data_rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 分割所有字段
            parts = line.split(',')
            
            if len(parts) < 3:
                print(parts)
                continue  # 跳过格式不正确的行
            
            # 前3列：时间戳, ID, DLC
            timestamp = parts[0]
            can_id = parts[1]
            dlc = parts[2]
            
            # 检查是否有方向标识（最后一列是R或T）
            has_flag = len(parts) > 3 and parts[-1].strip() in ['R', 'T']
            
            # 提取标签
            flag = parts[-1].strip() if has_flag else 'R'  # 默认为R
            
            # 数据字节部分：如果有标签，去掉最后一列；否则使用所有剩余列
            if has_flag:
                data_bytes = parts[3:-1]  # 去掉标签
            else:
                data_bytes = parts[3:] if len(parts) > 3 else []
            
            # 构建标准化的行：时间戳, ID, DLC, 数据字节1, 数据字节2, ..., 标签
            row = [timestamp, can_id, dlc] + data_bytes
            # 确保至少有11列（时间戳+ID+DLC+最多8字节数据）
            # 如果数据字节超过8个，只取前8个
            if len(row) >= 11:
                row = row[:11]
            else:
                # 如果不足11列，用空字符串填充
                while len(row) < 11:
                    row.append('')
            
            # 添加标签作为第12列（索引11）
            row.append(flag)
            
            data_rows.append(row)
    
    # 转换为DataFrame（12列：时间戳+ID+DLC+8字节数据+标签）
    file = pd.DataFrame(data_rows)
    columns = range(0, 12)
    file.columns = columns
    return file


def parse_mixed_format(file_path):
    """
    解析混合格式的Survival数据（FreeDriving）
    格式：时间戳,ID,DLC,数据字节（空格分隔）
    注意：FreeDriving数据没有标签，默认为'R'
    """
    data_rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 前3列用逗号分隔
            parts = line.split(',', 3)
            if len(parts) < 4:
                continue
            
            # 前3列：时间戳, ID, DLC
            timestamp = parts[0]
            can_id = parts[1]
            dlc = parts[2]
            data_bytes_str = parts[3]
            
            # 数据字节部分用空格分隔
            data_bytes = data_bytes_str.split()
            
            # 构建标准化的行：时间戳, ID, DLC, 数据字节1, 数据字节2, ...
            row = [timestamp, can_id, dlc] + data_bytes
            # 确保至少有11列（时间戳+ID+DLC+最多8字节数据）
            while len(row) < 11:
                row.append('')
            row = row[:11]  # 只取前11列
            
            # FreeDriving数据没有标签，默认为'R'（正常数据）
            row.append('R')
            
            data_rows.append(row)
    
    # 转换为DataFrame（12列：时间戳+ID+DLC+8字节数据+标签）
    file = pd.DataFrame(data_rows)
    columns = range(0, 12)
    file.columns = columns
    return file


# 对Survival Dataset数据进行预处理和特征提取
def survival_process_data(file_path, sliding_window):
    # 检测文件格式
    file_format, is_attack = detect_survival_file_format(file_path)
    
    if file_format is None:
        print("file_path不规范或无法识别格式")
        return None, None
    
    # 根据格式解析文件
    if file_format == 'csv_like':
        file = parse_csv_like_format(file_path)
    elif file_format == 'mixed':
        file = parse_mixed_format(file_path)
    else:
        print(f"未知的文件格式: {file_format}")
        return None, None

    data = clean_data(file, is_attack)
    data = data.values

    stride_time = sliding_window
    results = []
    labels = []
    stride_node = StrideNode(stride_time)
    size = int(30 / stride_time)
    car_queue = CarQueue(max_len=size, stride=stride_time)
    start_time = data[0][0]

    for index in range(len(data)):
        new_data = data[index]
        # t = time.time()
        if start_time + stride_time > new_data[0]:
            stride_node.add_data(new_data)
        else:
            car_queue.append(stride_node)
            if len(car_queue) == size:
                result, label = car_queue.get_result()
                results.extend(result)
                labels.extend(label)
            start_time += stride_time
            while start_time + stride_time < new_data[0]:
                stride_node = StrideNode(stride_time)
                car_queue.append(stride_node)
                start_time += stride_time
            stride_node = StrideNode(stride_time)
            stride_node.add_data(new_data)
            # print(time.time()-t)

        progress_bar(index, len(data))
    return results, labels


# 针对Survival Dataset的数据进行数据清洗
def clean_data(data, attack):
    # data = data.dropna(axis=0) // 该操作将有用信息全部删除了
    data = concatenate_columns(data)    
    result = data[[0, 1, 2, 'new_column', 11]].copy()  # 包含标签列（11）
    # 确保时间戳列（列0）是浮点数类型
    result[0] = pd.to_numeric(result[0], errors='coerce')
    if attack:
        # 确保DLC列（列2）是整数类型
        col2_plus_2 = data[2].astype(int) + 3
        result[3] = [data.loc[i, col] if col in data.columns else None
                     for i, col in enumerate(col2_plus_2)]
    result = result.dropna(axis=0)
    return result


def concatenate_columns(data):
    # 获取最大列数
    max_cols = data[2].astype(int) + 2
    max_col_overall = max_cols.max()

    # 创建一个空的结果Series
    result = pd.Series('', index=data.index)

    # 对每个可能的列进行向量化操作
    for col in range(3, max_col_overall + 1):
        if col in data.columns:
            mask = max_cols >= col
            result[mask] += data[col][mask].astype(str)

    data['new_column'] = result
    return data


def progress_bar(progress, total, bar_length=50):
    percent = progress / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\r|{bar}| {percent:.2%}', end='')

    # 当进度完成时换行
    if progress == total - 1:
        print()


if __name__ == '__main__':
    # 测试Survival数据集的不同格式
    # 基于脚本所在目录计算相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data', 'survival')
    
    file_path_flooding = os.path.join(data_dir, 'Flooding_dataset_concat.txt')
    file_path_freedriving = os.path.join(data_dir, 'FreeDriving_dataset_concat.txt')
    
    print("=" * 60)
    print("测试Flooding数据（CSV格式）:")
    print("=" * 60)
    feature, label = survival_process_data(file_path_flooding, 0.05)
    if feature is not None:
        print(f"\n特征数量: {len(feature)}, 标签数量: {len(label)}")
        print(f"\n特征类型: {type(feature)}")
        if isinstance(feature, list) and len(feature) > 0:
            print(f"特征元素类型: {type(feature[0])}")
            print(f"前5个特征示例:")
            for i in range(min(5, len(feature))):
                print(f"  [{i}]: {feature[i]}")
        elif isinstance(feature, np.ndarray):
            print(f"特征数组形状: {feature.shape}")
            print(f"特征数组数据类型: {feature.dtype}")
            print(f"前5个特征示例:")
            for i in range(min(5, len(feature))):
                print(f"  [{i}]: {feature[i]}")
        
        print(f"\n标签类型: {type(label)}")
        if isinstance(label, list) and len(label) > 0:
            print(f"标签元素类型: {type(label[0])}")
            print(f"前10个标签示例:")
            for i in range(min(10, len(label))):
                print(f"  [{i}]: {label[i]}")
            print(f"\n标签统计:")
            unique_labels, counts = np.unique(label, return_counts=True)
            for u, c in zip(unique_labels, counts):
                print(f"  标签 {u}: {c} 个 ({c/len(label)*100:.2f}%)")
        elif isinstance(label, np.ndarray):
            print(f"标签数组形状: {label.shape}")
            print(f"标签数组数据类型: {label.dtype}")
            print(f"前10个标签示例:")
            for i in range(min(10, len(label))):
                print(f"  [{i}]: {label[i]}")
            print(f"\n标签统计:")
            unique_labels, counts = np.unique(label, return_counts=True)
            for u, c in zip(unique_labels, counts):
                print(f"  标签 {u}: {c} 个 ({c/len(label)*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("测试FreeDriving数据（混合格式）:")
    print("=" * 60)
    feature, label = survival_process_data(file_path_freedriving, 0.05)
    if feature is not None:
        print(f"\n特征数量: {len(feature)}, 标签数量: {len(label)}")
        print(f"\n特征类型: {type(feature)}")
        if isinstance(feature, list) and len(feature) > 0:
            print(f"特征元素类型: {type(feature[0])}")
            print(f"前5个特征示例:")
            for i in range(min(5, len(feature))):
                print(f"  [{i}]: {feature[i]}")
        elif isinstance(feature, np.ndarray):
            print(f"特征数组形状: {feature.shape}")
            print(f"特征数组数据类型: {feature.dtype}")
            print(f"前5个特征示例:")
            for i in range(min(5, len(feature))):
                print(f"  [{i}]: {feature[i]}")
        
        print(f"\n标签类型: {type(label)}")
        if isinstance(label, list) and len(label) > 0:
            print(f"标签元素类型: {type(label[0])}")
            print(f"前10个标签示例:")
            for i in range(min(10, len(label))):
                print(f"  [{i}]: {label[i]}")
            print(f"\n标签统计:")
            unique_labels, counts = np.unique(label, return_counts=True)
            for u, c in zip(unique_labels, counts):
                print(f"  标签 {u}: {c} 个 ({c/len(label)*100:.2f}%)")
        elif isinstance(label, np.ndarray):
            print(f"标签数组形状: {label.shape}")
            print(f"标签数组数据类型: {label.dtype}")
            print(f"前10个标签示例:")
            for i in range(min(10, len(label))):
                print(f"  [{i}]: {label[i]}")
            print(f"\n标签统计:")
            unique_labels, counts = np.unique(label, return_counts=True)
            for u, c in zip(unique_labels, counts):
                print(f"  标签 {u}: {c} 个 ({c/len(label)*100:.2f}%)")
