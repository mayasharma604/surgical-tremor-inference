import torch
import torch.nn as nn

class IntentLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)  # output is (x, y)

    def forward(self, x):
        # x shape: (batch, 2)
        x = x.unsqueeze(1)  # (batch, seq_len=1, 2)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)
