import torch
import torch.nn as nn
from collections import OrderedDict

class ConvBlock(nn.Module):
    def __init__(self, kernel_size, max_pool_size, in_channel, out_channel):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=0),
            #nn.BatchNorm2d(out_channel),
            #nn.ELU(),
            #nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=max_pool_size)
        )
    def forward(self, X):
        #out = self.convblock(X)
        return self.convblock(X)

class CnnLstm(nn.Module):

    def __init__(self, dummy_input, device, kernel_sizes = [(1, 1)],
                    max_pool_sizes = [(1, 1)],
                    filter_sizes = [15], hidden_size = 50):
        
        """CNN-LSTM model
        Dummy input dimension : N, T, C
        """
        super().__init__()
        N, T, C = dummy_input.shape 
        cnn_dummy_input = dummy_input.unsqueeze(-1).swapaxes(1, 2)
        
        sequantial_block = OrderedDict()
        for i, (kernel_size, max_pool_size, filter_size) in enumerate(zip(kernel_sizes, max_pool_sizes, filter_sizes)):
            if i == 0:
                sequantial_block[f'ConvBlock-{i}'] = ConvBlock(
                    kernel_size, max_pool_size, in_channel=C, out_channel=filter_size
                )
                continue
            sequantial_block[f'ConvBlock-{i}'] = ConvBlock(
                kernel_size, max_pool_size, in_channel=filter_sizes[i - 1], out_channel=filter_size
            )
        
        self.sequantial_block = nn.Sequential(sequantial_block)
        self.sequantial_block.to(device)
        
        N, C_out, T, _ = self.sequantial_block(cnn_dummy_input).shape
        print(N, C_out, T, 1)

        self.lstm = nn.LSTM(input_size=C_out, hidden_size=hidden_size, num_layers=1, batch_first=True)
        

        self.linear_block = nn.Sequential(
            nn.Linear(hidden_size ,1) 
        )
        self.linear_block.to(device)

    def forward(self, X):
        """X has shape N, T, C"""
        reshaped_X = X.swapaxes(1, 2).unsqueeze(-1)
        out = self.sequantial_block(reshaped_X)
        out = out.swapaxes(1, 2)
        out = out.squeeze(-1)
        
        out, _= self.lstm(out)
        out = out[:, -1, :]
        out = self.linear_block(out)
        return out
        

            
        