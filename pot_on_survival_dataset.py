import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from NLT.NLT_main import likelihood_transformation
from data_processing.timeseries_dataset import loading_survival_dataset, dataClassifier
from model_test.CustomClass import SimpleConcatDataset
from model_test.Custom_split import my_random_split
from model_test.grid_research import grid_research
from model.LSTM import LSTMAutoencoder, train_model

if __name__ == '__main__':
    # model setting
    batch_size = 100
    input_size = 1
    num_layers = 2
    hidden_size = 50
    epoch = 4000
    window_size = 10
    learning_rate = 0.0005
    sliding_window = 0.05

    # 1. 先检查GPU是否可用
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_file = 'car_hacking_model.pt'
    transfer = likelihood_transformation()

    # 加载数据集
    data_set = loading_survival_dataset(window_size, sliding_window, transfer, 'all')
    normal_dataset, attack_dataset = dataClassifier(data_set)

    # 数据集划分
    train_size = int(0.8 * len(normal_dataset))
    test_size = len(normal_dataset) - train_size
    train_normal_dataset, test_norma_dataset = my_random_split(normal_dataset, [train_size, test_size])

    train_size = int(0.8 * len(attack_dataset))
    test_size = len(attack_dataset) - train_size
    train_attack_dataset, test_attack_dataset = my_random_split(attack_dataset, [train_size, test_size])
    test_dataset = SimpleConcatDataset([test_norma_dataset, test_attack_dataset])

    # 2. 创建模型并明确移到GPU
    model = LSTMAutoencoder(input_size, hidden_size, device)
    model = model.to(device)  # 重要：将模型移到GPU

    print("training model")
    train_loader = DataLoader(train_normal_dataset, batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. 检查train_model函数内部是否处理了数据转移
    model = train_model(model, train_loader, criterion, optimizer, epoch, device)

    # 4. 保存模型
    torch.save(model.state_dict(), model_file)
    print('training Done')

    # 5. 测试时也要确保模型和数据在GPU上
    model.eval()
    with torch.no_grad():
        grid = grid_research(test_dataset, model)

    grid.append(transfer.get_global_max())
    file_path = 'grid.json'
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(grid, f, ensure_ascii=False, indent=2)
    print(f"参数已保存 {grid}")