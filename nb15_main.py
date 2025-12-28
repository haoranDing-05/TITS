from torch import nn, optim
from torch.utils.data import random_split, DataLoader, ConcatDataset
import torch
from data_processing.timeseries_dataset import TimeSeriesDataset
from model.LSTM import LSTMAutoencoder, train_model

# setting
batch_size = 50
hidden_size = 100
epoch = 400
window_size = 10
input_size = 1
num_layers = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NB15_1 = "NB15_1"
NB15_2 = "NB15_2"
NB15_3 = "NB15_3"
NB15_4 = "NB15_4"

nb15_1 = TimeSeriesDataset(NB15_1, 10)
nb15_2 = TimeSeriesDataset(NB15_2, 10)
nb15_3 = TimeSeriesDataset(NB15_3, 10)
nb15_4 = TimeSeriesDataset(NB15_4, 10)

nb15_dataset = ConcatDataset([nb15_1, nb15_2, nb15_3, nb15_4])

train_size = int(0.8 * len(nb15_dataset))
test_size = len(nb15_dataset) - train_size
# 使用 random_split 函数将数据集分割成训练集和测试集
train_dataset, test_dataset = random_split(nb15_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

model = LSTMAutoencoder(input_size, hidden_size, num_layers)
criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
train_model(model, train_loader, test_loader, criterion, optimizer, epoch, device)
