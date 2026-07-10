import torch.nn as nn
from torch.nn import functional as F


class LSTMStack(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(LSTMStack, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.lstm = nn.LSTM(
            in_shape, 64, 2, batch_first=True, dropout=0.1
        )  
        self.linear1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, out_shape)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, targets):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.linear2(x)
        if targets is None:
            return logits

        loss = F.mse_loss(logits, targets)
        return logits, loss
