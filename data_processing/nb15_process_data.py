import time

import pandas as pd
import torch
import numpy as np


#file_path = r".\unsw-nb15\versions\1\UNSW-NB15_1.csv"


def nb15_process_data(file_path):
    # 读取 CSV 文件
    file_csv = pd.read_csv(file_path, header=None, low_memory=False)
    # 选择需要的列
    file = file_csv.iloc[:, [0, 2, 6, 7, 16, 28]]

    file = file.dropna(axis=0)
    # 初始化结果列表
    result = []
    # 获取起始时间和结束时间
    t = file.loc[0, 28]
    t_end = list(file.loc[:, 28])[-1]
    # 循环处理每个时间窗口
    while True:
        # 创建时间窗口的掩码
        mask = (file.loc[:, 28] >= t) & (file.loc[:, 28] < t + 3)
        # 根据掩码选择数据
        temp_data = file.loc[mask, [0, 2, 6, 7, 16]]

        # 进行分组求和操作
        result1 = temp_data.groupby(0)[7].sum()
        result2 = temp_data.groupby(0)[6].sum()
        result3 = temp_data.groupby([0, 2])[7].sum()
        result4 = temp_data.groupby([0, 2])[6].sum()
        result5 = temp_data.groupby([0, 2])[16].sum()

        # 计算统计量
        temp_res = [np.mean(result1), np.std(result1),
                    np.mean(result2), np.std(result2),
                    np.mean(result3), np.std(result3),
                    np.mean(result4), np.std(result4),
                    np.mean(result5), np.std(result5)]

        # 将当前时间窗口的统计结果添加到结果列表中
        result.append(temp_res)

        # 更新时间
        t += 3

        # 如果时间超过结束时间，则退出循环
        if t > t_end:
            break
    final_result = np.array(result)
    print('read over')
    print(len(final_result))
    return final_result

#调用函数获得最终结果
#final=process_data(file_path)
#print(type(final))
