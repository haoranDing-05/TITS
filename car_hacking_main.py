import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from NLT.NLT_main import likelihood_transformation
from data_processing.timeseries_dataset import loading_car_hacking, dataClassifier
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
    epoch = 500
    window_size = 10
    learning_rate = 0.001
    sliding_window = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_file = 'car_hacking_model.pt'

    transfer = likelihood_transformation()

    # train_dataset = loading_car_hacking(window_size, sliding_window, transfer, 'train')
    # normal_dataset, attack_dataset = dataClassifier(train_dataset)
    #
    # val_dataset = loading_car_hacking(window_size, sliding_window, transfer, 'test')

    data_set = loading_car_hacking(window_size, sliding_window, transfer, 'all')

    normal_dataset, attack_dataset = dataClassifier(data_set)

    train_size = int(0.8 * len(normal_dataset))
    test_size = len(normal_dataset) - train_size
    train_normal_dataset, test_norma_dataset = my_random_split(normal_dataset, [train_size, test_size])

    train_size = int(0.8 * len(attack_dataset))
    test_size = len(attack_dataset) - train_size
    train_attack_dataset, test_attack_dataset = my_random_split(attack_dataset, [train_size, test_size])
    test_dataset = SimpleConcatDataset([test_norma_dataset, test_attack_dataset])

    model = LSTMAutoencoder(input_size, hidden_size, device)
    print("training model")
    train_loader = DataLoader(normal_dataset, batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = train_model(model, train_loader, criterion, optimizer, epoch, device)
    torch.save(model.state_dict(), model_file)
    print('training Done')

    grid = grid_research(test_dataset, model)
    grid.append(transfer.get_global_max())
    file_path = 'grid.json'
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(grid, f, ensure_ascii=False, indent=2)
    print(f"参数已保存 {grid}")
