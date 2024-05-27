import torch
import torch.nn as nn
import math

from GeneralModel import GeneralModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class STransformer(GeneralModel):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout, T=1000, max_len=5000):
        super().__init__("STransformer")
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_layers, enable_nested_tensor=False
        )
        self.linear = nn.Linear(T * d_model, 1)  # Flatten and map to output

    def forward(self, x):
        x = self.input_proj(x)  # Project input to d_model
        x = x.permute(1, 0, 2)  # Transformer expects (T, N, C)
        x = self.positional_encoding(x)
        mask = self.generate_square_subsequent_mask(x.size(0))
        x = self.transformer_encoder(x, mask)
        x = x.permute(1, 0, 2).contiguous()  # Back to (N, T, C)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.linear(x)
        return x
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask