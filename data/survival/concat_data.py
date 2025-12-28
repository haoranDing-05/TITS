import os
import glob
import pandas as pd
from pathlib import Path


def get_file_start_time(file_path):
    """获取文件的第一行时间戳"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                # 提取时间戳（第一列，逗号分隔）
                timestamp = float(first_line.split(',')[0])
                return timestamp
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
    return None


def concat_flooding_data(output_file='Flooding_dataset_concat.txt'):
    """
    合并所有Flooding类型的数据文件，按照起始时间从早到晚排序
    """
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent
    
    # 查找所有Flooding文件
    flooding_files = glob.glob(str(current_dir / 'Flooding_dataset_*.txt'))
    
    if not flooding_files:
        print("未找到Flooding类型的数据文件")
        return
    
    # 获取每个文件的起始时间戳
    file_times = []
    for file_path in flooding_files:
        start_time = get_file_start_time(file_path)
        if start_time is not None:
            file_times.append((start_time, file_path))
    
    # 按时间戳排序
    file_times.sort(key=lambda x: x[0])
    
    print(f"找到 {len(file_times)} 个Flooding文件，按时间顺序合并:")
    for start_time, file_path in file_times:
        print(f"  {os.path.basename(file_path)}: {start_time}")
    
    # 合并文件
    output_path = current_dir / output_file
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for start_time, file_path in file_times:
            print(f"正在合并: {os.path.basename(file_path)}")
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
    
    print(f"合并完成，输出文件: {output_path}")


def concat_freedriving_data(output_file='FreeDriving_dataset_concat.txt'):
    """
    合并所有FreeDriving类型的数据文件，按照起始时间从早到晚排序
    """
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent
    
    # 查找所有FreeDriving文件
    freedriving_files = glob.glob(str(current_dir / 'FreeDrivingData_*.txt'))
    
    if not freedriving_files:
        print("未找到FreeDriving类型的数据文件")
        return
    
    # 获取每个文件的起始时间戳
    file_times = []
    for file_path in freedriving_files:
        start_time = get_file_start_time(file_path)
        if start_time is not None:
            file_times.append((start_time, file_path))
    
    # 按时间戳排序
    file_times.sort(key=lambda x: x[0])
    
    print(f"找到 {len(file_times)} 个FreeDriving文件，按时间顺序合并:")
    for start_time, file_path in file_times:
        print(f"  {os.path.basename(file_path)}: {start_time}")
    
    # 合并文件
    output_path = current_dir / output_file
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for start_time, file_path in file_times:
            print(f"正在合并: {os.path.basename(file_path)}")
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
    
    print(f"合并完成，输出文件: {output_path}")


def concat_fuzzy_data(output_file='Fuzzy_dataset_concat.txt'):
    """
    合并所有Fuzzy类型的数据文件，按照起始时间从早到晚排序
    """
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent
    
    # 查找所有Fuzzy文件
    fuzzy_files = glob.glob(str(current_dir / 'Fuzzy_dataset_*.txt'))
    
    if not fuzzy_files:
        print("未找到Fuzzy类型的数据文件")
        return
    
    # 获取每个文件的起始时间戳
    file_times = []
    for file_path in fuzzy_files:
        start_time = get_file_start_time(file_path)
        if start_time is not None:
            file_times.append((start_time, file_path))
    
    # 按时间戳排序
    file_times.sort(key=lambda x: x[0])
    
    print(f"找到 {len(file_times)} 个Fuzzy文件，按时间顺序合并:")
    for start_time, file_path in file_times:
        print(f"  {os.path.basename(file_path)}: {start_time}")
    
    # 合并文件
    output_path = current_dir / output_file
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for start_time, file_path in file_times:
            print(f"正在合并: {os.path.basename(file_path)}")
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
    
    print(f"合并完成，输出文件: {output_path}")


def concat_malfunction_data(output_file='Malfunction_dataset_concat.txt'):
    """
    合并所有Malfunction类型的数据文件，按照起始时间从早到晚排序
    """
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent
    
    # 查找所有Malfunction文件
    malfunction_files = glob.glob(str(current_dir / 'Malfunction*.txt'))
    
    if not malfunction_files:
        print("未找到Malfunction类型的数据文件")
        return
    
    # 获取每个文件的起始时间戳
    file_times = []
    for file_path in malfunction_files:
        start_time = get_file_start_time(file_path)
        if start_time is not None:
            file_times.append((start_time, file_path))
    
    # 按时间戳排序
    file_times.sort(key=lambda x: x[0])
    
    print(f"找到 {len(file_times)} 个Malfunction文件，按时间顺序合并:")
    for start_time, file_path in file_times:
        print(f"  {os.path.basename(file_path)}: {start_time}")
    
    # 合并文件
    output_path = current_dir / output_file
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for start_time, file_path in file_times:
            print(f"正在合并: {os.path.basename(file_path)}")
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
    
    print(f"合并完成，输出文件: {output_path}")


if __name__ == '__main__':
    # 可以单独运行某个函数，或者全部运行
    print("=" * 50)
    print("合并Flooding数据")
    print("=" * 50)
    concat_flooding_data()
    
    print("\n" + "=" * 50)
    print("合并FreeDriving数据")
    print("=" * 50)
    concat_freedriving_data()
    
    print("\n" + "=" * 50)
    print("合并Fuzzy数据")
    print("=" * 50)
    concat_fuzzy_data()
    
    print("\n" + "=" * 50)
    print("合并Malfunction数据")
    print("=" * 50)
    concat_malfunction_data()
