import torch
import torch.nn as nn 
from GeneralModel import GeneralModel

class LSTM(GeneralModel):
    """Vanilla LSTM"""
    def __init__(self, input_dim : int, hidden_size : int, device,
                dropout = 0.5, num_layers = 2, model_name = "LSTM", out_size = 1):
        super().__init__(model_name)

        self.lstm = nn.LSTM(input_size = input_dim, 
                hidden_size = hidden_size,
                batch_first=True,
                dropout=dropout,
                num_layers=num_layers, device=device)

        self.linear = nn.Linear(hidden_size, out_size, device=device)
        
    def forward(self, X):
        out, _ = self.lstm(X)
        out = self.linear(out[:, -1, :])
        return out
