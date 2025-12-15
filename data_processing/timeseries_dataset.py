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
    
    normal_run_path = rf'..\Car-Hacking Dataset 2-8\{mode}\normal_run_data.txt'
    DoS_dataset_path = rf'..\Car-Hacking Dataset 2-8\{mode}\DoS_dataset.csv'
    Fuzzy_dataset_path = rf'..\Car-Hacking Dataset 2-8\{mode}\Fuzzy_dataset.csv'
    RPM_dataset_path = rf'..\Car-Hacking Dataset 2-8\{mode}\RPM_dataset.csv'
    gear_dataset_path = rf'..\Car-Hacking Dataset 2-8\{mode}\gear_dataset.csv'

    if mode == 'all':
        normal_run_path = rf'.\Car-Hacking Dataset\normal_run_data\normal_run_data.txt'
        DoS_dataset_path = rf'.\Car-Hacking Dataset\DoS_dataset.csv'
        Fuzzy_dataset_path = rf'.\Car-Hacking Dataset\Fuzzy_dataset.csv'
        RPM_dataset_path = rf'.\Car-Hacking Dataset\RPM_dataset.csv'
        gear_dataset_path = rf'.\Car-Hacking Dataset\gear_dataset.csv'

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

def loading_challenge_dataset(window_size, sliding_window, transfer, mode):
    from data_processing.challenge_process_data import challenge_process_data
    import os
    
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data', 'Car_Hacking_Challenge_Dataset_rev20Mar2021')
    
    # 正常数据路径
    normal_path = os.path.join(data_dir, 'normal_combined_pre_train_D0S0.csv')
    
    # 攻击数据路径
    attack_path = os.path.join(data_dir, 'attack_combined.csv')
    
    # 根据mode选择文件
    if mode == 'all':
        # 使用所有数据
        file_path = [normal_path, attack_path]
    elif mode == 'normal':
        # 只使用正常数据
        file_path = [normal_path]
    elif mode == 'attack':
        # 只使用攻击数据
        file_path = [attack_path]
    else:
        # 默认使用所有数据
        file_path = [normal_path, attack_path]
    
    challenge_dataset = TimeSeriesDataset(file_path, window_size, sliding_window, transfer, challenge_process_data)
    
    return challenge_dataset    

def dataClassifier(dataset):
    # for i in range(len(dataset)):
    #     print(dataset.get_test_sample(i)[1])
    indices_0 = [i for i in range(len(dataset)) if dataset.get_test_sample(i)[1] == 0]
    indices_1 = [i for i in range(len(dataset)) if dataset.get_test_sample(i)[1] == 1]
    # 步骤2：创建子集
    attack_dataset = SimpleSubset(dataset, indices_0)
    normal_dataset = SimpleSubset(dataset, indices_1)

    return normal_dataset, attack_dataset
