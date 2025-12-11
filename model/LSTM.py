import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    # 将 silence_threshold 放在最后并设置默认值，这样 car_hacking_main.py 中
    # LSTMAutoencoder(input_size, hidden_size, device) 的调用依然有效
    def __init__(self, input_size, hidden_size, device, silence_threshold=1e-3):
        super(LSTMAutoencoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.silence_threshold = silence_threshold

        # 编码器和解码器
        self.encoder = CustomLSTMLayer(input_size, hidden_size, silence_threshold)
        self.decoder = CustomLSTMLayer(hidden_size, hidden_size, silence_threshold)
        self.reconstruct = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 初始化隐藏状态
        h_enc, c_enc, c_tidle_enc = self.init_hidden(batch_size)

        # 编码器前向传播
        for t in range(seq_len):
            input_t = x[:, t, :]
            # 这里的 c_tidle_enc 只是为了保持接口兼容，内部计算已不再依赖上一时刻的 candidate
            h_enc, c_enc, c_tidle_enc = self.encoder(input_t, h_enc, c_enc, c_tidle_enc)

        # 解码器前向传播
        decoder_outputs = []
        h_dec = h_enc
        c_dec = c_enc
        c_tidle_dec = c_tidle_enc  # 继承状态
        decoder_input = h_dec

        for t in range(seq_len):
            h_dec, c_dec, c_tidle_dec = self.decoder(decoder_input, h_dec, c_dec, c_tidle_dec)
            output_t = self.reconstruct(h_dec)
            decoder_outputs.append(output_t)
            # 下一步输入为当前隐状态
            decoder_input = h_dec

        # 返回完整重构序列
        reconstructed = torch.stack(decoder_outputs, dim=1)
        return reconstructed

    def init_hidden(self, batch_size):
        h = torch.zeros(batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(batch_size, self.hidden_size, device=self.device)
        c_tidle = torch.zeros(batch_size, self.hidden_size, device=self.device)
        return h, c, c_tidle


class CustomLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, silence_threshold=1e-3):
        super(CustomLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.silence_threshold = silence_threshold  # 新增：静默阈值

        # --- 保持原代码的显式权重定义 ---
        # 遗忘门参数
        self.W_f = nn.Linear(input_size, hidden_size)
        self.V_f = nn.Linear(hidden_size, hidden_size)
        self.b_f = nn.Parameter(torch.ones(hidden_size))  # 保持初始化为1

        # 输入门参数
        self.W_i = nn.Linear(input_size, hidden_size)
        self.V_i = nn.Linear(hidden_size, hidden_size)
        self.b_i = nn.Parameter(torch.zeros(hidden_size))

        # 输出门参数
        self.W_o = nn.Linear(input_size, hidden_size)
        self.V_o = nn.Linear(hidden_size, hidden_size)
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

        # 细胞状态参数
        self.W_c = nn.Linear(input_size, hidden_size)
        self.V_c = nn.Linear(hidden_size, hidden_size)
        self.b_c = nn.Parameter(torch.zeros(hidden_size))

        self._initialize_weights()

    def forward(self, x, h_prev, c_prev, c_prev_tilde_dummy):
        """
        x: 当前输入
        h_prev: 上一时刻隐状态
        c_prev: 上一时刻细胞状态
        c_prev_tilde_dummy: 上一时刻的候选状态 (为了兼容原代码接口保留，但在鲁棒逻辑中不参与计算)
        """

        # 1. 计算门控 (保持原代码风格)
        f_t = torch.sigmoid(self.W_f(x) + self.V_f(h_prev) + self.b_f)
        i_t = torch.sigmoid(self.W_i(x) + self.V_i(h_prev) + self.b_i)
        o_t = torch.sigmoid(self.W_o(x) + self.V_o(h_prev) + self.b_o)

        # 当前时刻的候选状态 (Corrected: 使用当前 x 和 h_prev 计算，而非使用 c_prev_tilde)
        # 这就是鲁棒机制中的 C_tilde_t
        c_tilde_current = torch.tanh(self.W_c(x) + self.V_c(h_prev) + self.b_c)

        # 2. 计算静默指示函数 (新增逻辑：来自第二个代码)
        # 计算输入的L2范数
        input_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        # active_mask: 大于阈值为 1 (活跃)，否则为 0 (静默)
        active_mask = (input_norm > self.silence_threshold).float()

        # 3. 鲁棒细胞状态更新 (核心修改)
        # 公式: C_t = (f_t * C_{t-1}) + (i_t * C_{t-1}) + Mask * (i_t * C_tilde_t)

        term1 = f_t * c_prev  # 基础遗忘保持
        term2 = i_t * c_prev  # 额外保持 (鲁棒机制特征)
        term3 = active_mask * (i_t * c_tilde_current)  # 仅在活跃时写入新信息

        c_next = term1 + term2 + term3

        # 4. 计算隐藏状态
        h_next = o_t * torch.tanh(c_next)

        # 返回 h_next, c_next, 以及当前的 c_tilde (以便循环中继续传递，保持格式一致)
        return h_next, c_next, c_tilde_current

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=1.0)
            elif 'bias' in name:
                if 'b_f' in name:
                    nn.init.constant_(param, 1.0)
                else:
                    nn.init.constant_(param, 0.0)


# --- 必须将 train_model 放在类外面，并取消注释，否则无法被 import ---
def train_model(model, train_loader, criterion, optimizer, epochs, device):
    model.to(device)

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        batch_count = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        avg_train_loss = total_loss / batch_count
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.8f}")
    return model