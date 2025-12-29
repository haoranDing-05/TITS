import numpy as np
import torch
from torch.utils.data import Dataset

from model_test.CustomClass import SimpleSubset


# 将特征提取后的数据转变为可供训练的Dataset类
class TimeSeriesDataset(Dataset):
    def __init__(self, file_path, window_size, sliding_window, transfer, process_func):
        features = None
        label = []
        for path in file_path:
            print(f"loading {path}")
            if features is None:
                features, label = process_func(path, sliding_window)
            else:
                feature, labels = process_func(path, sliding_window)
                features = np.concatenate([features, feature], axis=0)
                label = np.concatenate([label, labels], axis=0)
        features = np.array(features)
        label = np.array(label)

        features = transfer.out(features)
        
        # 数据预处理可视化
        from matplotlib import pyplot as plt
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        plt.plot([i for i in range(len(features))], features, label='num')
        plt.plot([i for i in range(len(label))], label, label='label')
        plt.show()

        features = features.reshape(len(features), 1)

        # 滑动窗口为10
        dataset_x, dataset_y = [], []
        dataset_label = []
        index = 0
        while index < len(features) - 10:
            _x = features[index:(index + window_size)]
            dataset_x.append(_x)
            dataset_y.append(features[index + window_size])
            # label中1是正常0是攻击
            has_attack = False
            # 检查窗口内的每个标签
            for j in range(window_size):
                if label[index + j] == 0:
                    has_attack = True
                    break
            dataset_label.append(0 if has_attack else 1)  # 0=攻击，1=正常
            index += 10

        # 转换为 PyTorch 张量
        self.X = torch.tensor(dataset_x, dtype=torch.float32)
        self.y = self.X
        self.label = torch.tensor(dataset_label, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    #用于返回（X,label） 用来与判断出的类别比对
    def get_test_sample(self, index=None):  #我想实现默认不输入时返回整个列表，但是不传入index时总是报错
        if index is None:
            return self.X, self.label
        return self.X[index], self.label[index]


def loading_car_hacking(window_size, sliding_window, transfer, mode):
    from data_processing.car_hacking_process_data import car_hacking_process_data
    import os

    # 1. 获取项目根目录 (无论你在哪里运行，这行都能找到 TITS 文件夹的绝对路径)
    # 假设 timeseries_dataset.py 在 TITS/data_processing/ 下，需要往上走两层或者根据实际位置调整
    # 如果这个文件在根目录，用 os.path.dirname(os.path.abspath(__file__)) 即可
    # 如果文件在 data_processing 文件夹内：
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)  # 假设上一级就是根目录

    # 如果这个文件本身就在根目录（看你上传的结构似乎是的），那就直接用：
    # project_root = os.path.dirname(os.path.abspath(__file__))

    # 2. 拼接正确路径
    dataset_folder = os.path.join(project_root, 'Car-Hacking Dataset')

    if mode == 'all':
        normal_run_path = os.path.join(dataset_folder, 'normal_run_data', 'normal_run_data.txt')
        DoS_dataset_path = os.path.join(dataset_folder, 'DoS_dataset.csv')
        Fuzzy_dataset_path = os.path.join(dataset_folder, 'Fuzzy_dataset.csv')
        RPM_dataset_path = os.path.join(dataset_folder, 'RPM_dataset.csv')
        gear_dataset_path = os.path.join(dataset_folder, 'gear_dataset.csv')
    else:
        # 处理其他 mode 的逻辑，建议也统一用 os.path.join
        pass

    file_path = [normal_run_path, DoS_dataset_path, Fuzzy_dataset_path, RPM_dataset_path, gear_dataset_path]
    car_hacking_dataset = TimeSeriesDataset(file_path, window_size, sliding_window, transfer, car_hacking_process_data)

    return car_hacking_dataset

def loading_survival_dataset(window_size, sliding_window, transfer, mode):
    from data_processing.survival_process_data import survival_process_data
    import os
    
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data', 'survival')
    
    # 正常数据路径
    freedriving_path = os.path.join(data_dir, 'FreeDriving_dataset_concat.txt')
    
    # 攻击数据路径
    flooding_path = os.path.join(data_dir, 'Flooding_dataset_concat.txt')
    fuzzy_path = os.path.join(data_dir, 'Fuzzy_dataset_concat.txt')
    malfunction_path = os.path.join(data_dir, 'Malfunction_dataset_concat.txt')
    
    # 根据mode选择文件
    if mode == 'all':
        # 使用所有数据
        file_path = [freedriving_path, flooding_path, fuzzy_path, malfunction_path]
    elif mode == 'normal':
        # 只使用正常数据
        file_path = [freedriving_path]
    elif mode == 'attack':
        # 只使用攻击数据
        file_path = [flooding_path, fuzzy_path, malfunction_path]
    else:
        # 默认使用所有数据
        file_path = [freedriving_path, flooding_path, fuzzy_path, malfunction_path]
    
    survival_dataset = TimeSeriesDataset(file_path, window_size, sliding_window, transfer, survival_process_data)
    
    return survival_dataset

def dataClassifier(dataset):
    # for i in range(len(dataset)):
    #     print(dataset.get_test_sample(i)[1])
    indices_0 = [i for i in range(len(dataset)) if dataset.get_test_sample(i)[1] == 0]
    indices_1 = [i for i in range(len(dataset)) if dataset.get_test_sample(i)[1] == 1]
    # 步骤2：创建子集
    attack_dataset = SimpleSubset(dataset, indices_0)
    normal_dataset = SimpleSubset(dataset, indices_1)

    return normal_dataset, attack_dataset


if __name__ == '__main__':
    from NLT.NLT_main import likelihood_transformation
    window_size = 10
    sliding_window = 0.05
    transfer = likelihood_transformation()
    ch_data = loading_car_hacking(window_size, sliding_window, transfer, 'all')
    print(type(ch_data))