from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from torch.utils.data import Subset


#在ConcatDataset基础上定义新类，实现了获得形如(X,label)的数据
class SimpleConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.sub_datasets = datasets  # 显式保存子数据集列表

    def get_test_sample(self, index):  #我想实现默认不输入时返回整个列表，但是不传入index时总是报错
        # 遍历子数据集，找到 idx 对应的那个
        for dataset in self.sub_datasets:
            if index < len(dataset):
                return dataset.get_test_sample(index)  # 调用子数据集的方法
            index -= len(dataset)
        raise IndexError("索引超出范围")


#在Dataset基础上定义新类，#没实现#从dataloader获得形如(X,label)的数据
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx=None):  #我想实现默认不输入时返回整个列表，但是不传入index时总是报错
        if (idx == None):
            return self.data, self.labels
        else:
            # 可以根据需求修改返回的内容
            sample = self.data[idx]
            label = self.labels[idx]
            return sample, label


#在Subset基础上定义新类，实现了获得形如(X,label)的数据
class SimpleSubset(Subset):
    def get_test_sample(self, subset_idx=None):  #我想实现默认不输入时返回整个列表，但是不传入index时总是报错
        if subset_idx == None:
            return self.dataset.get_test_sample()
        else:
            original_idx = self.indices[subset_idx]
            return self.dataset.get_test_sample(original_idx)
