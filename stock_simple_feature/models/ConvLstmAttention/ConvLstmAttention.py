import torch
import torch.nn as nn
from collections import OrderedDict

class ConvBlock(nn.Module):

    def __init__(self, input_channel, filter_sizes = [15], maxpool_size = [(1, 1)]):
        # Asssume X has shape of N, T, C
        super().__init__()

        conv_list = OrderedDict()
        for i, (filter_size, maxpool_size) in enumerate(zip(filter_sizes, maxpool_size)):
            if i == 0:
                conv_list[f"ConvBlock-{i}"] = nn.Sequential(
                    nn.Conv2d(in_channels=input_channel, out_channels=filter_size, kernel_size=1, padding=0),
                    nn.BatchNorm2d(num_features=filter_size),
                    nn.MaxPool2d(kernel_size=maxpool_size))

            else:
                conv_list[f"ConvBlock-{i}"] = nn.Sequential(
                    nn.Conv2d(in_channels=filter_sizes[i-1], out_channels=filter_size, kernel_size=1, padding=0),
                    nn.BatchNorm2d(num_features=filter_size),
                    nn.MaxPool2d(kernel_size=maxpool_size))
        self.convblocks = nn.Sequential(conv_list)
    
    def forward(self, X):
        # X has shape of N, T, C
        X = X.swapaxes(1, 2).unsqueeze(-1) # N, C, T, 1
        out = self.convblocks(X)  # N, Channels, T, 1
        out = out.swapaxes(1, 2).squeeze(-1) # N, T, C
        return out

        
        

class ConvLstmAttention(nn.Module):

    def __init__(self, dummy_input, device):

        super().__init__()

        N, T, C = dummy_input.shape

        self.convblock = ConvBlock(C)
        self.convblock.to(device)

        conv_out = self.convblock(dummy_input)
        N, T, C_conv_out = conv_out.shape 

        
        self.lstm = nn.LSTM(
            input_size = C_conv_out, hidden_size=30, batch_first=True
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=30, num_heads=1, batch_first=True, bias=False
        )
        self.layer_norm = nn.LayerNorm(
            normalized_shape=30
        )
        self.fc = nn.Linear(30, 1)

    def forward(self, X):
        conv_out = self.convblock(X) # N, T, C
        lstm_out, _ = self.lstm(conv_out) # N, T, Hidden
        N, T, hidden_size = lstm_out.shape

        atten_mask = torch.triu(torch.ones(T, T, device=X.device), diagonal=1).bool()
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, attn_mask=atten_mask) # N, T, C
        attn_out = self.layer_norm(attn_out)
        fc_input = attn_out[:, -1, :] # N, Hidden
        return self.fc(fc_input) # N, 1
    
    def forward_attention(self, X):
        conv_out = self.convblock(X) # N, T, C
        lstm_out, _ = self.lstm(conv_out) # N, T, Hidden
        N, T, hidden_size = lstm_out.shape

        atten_mask = torch.triu(torch.ones(T, T, device=X.device), diagonal=1).bool()
        attn_out, attention_weight = self.attention(lstm_out, lstm_out, lstm_out, attn_mask=atten_mask) # N, T, C
        attn_out = self.layer_norm(attn_out)
        fc_input = attn_out[:, -1, :] # N, Hidden
        return self.fc(fc_input), attention_weight # N, 1
        