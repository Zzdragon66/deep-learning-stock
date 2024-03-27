import torch
import torch.nn as nn
from collections import OrderedDict

class ConvBlock(nn.Module):
    def __init__(self, kernel_size, max_pool_size, in_channel, out_channel):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding="same"),
            nn.BatchNorm2d(out_channel),
            #nn.ELU(),
            #nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=max_pool_size)
        )
    def forward(self, X):
        out = self.convblock(X)
        #print(out.shape)
        return self.convblock(X)

class CNN(nn.Module):

    def __init__(self, dummy_input, device, kernel_sizes = [(7, 5), (5, 3), (3, 1)],
                    max_pool_sizes = [(7, 5), (5, 3), (3, 1)],
                    filter_sizes = [15, 30, 60]):
        """Assume the dummy input shape has (N, T, C)"""
        super().__init__()

        N, T, C = dummy_input.shape
        reshaped_dummy_input = dummy_input.unsqueeze(1)

        
        sequantial_block = OrderedDict()
        for i, (kernel_size, max_pool_size, filter_size) in enumerate(zip(kernel_sizes, max_pool_sizes, filter_sizes)):
            if i == 0:
                sequantial_block[f'ConvBlock-{i}'] = ConvBlock(
                    kernel_size, max_pool_size, in_channel=1, out_channel=filter_size
                )
                continue
            sequantial_block[f'ConvBlock-{i}'] = ConvBlock(
                kernel_size, max_pool_size, in_channel=filter_sizes[i - 1], out_channel=filter_size
            )
        sequantial_block["flatten"] = nn.Flatten()

        self.sequantial_block = nn.Sequential(sequantial_block)
        self.sequantial_block.to(device)
        
        N, linear_input_shape = self.sequantial_block(reshaped_dummy_input).shape

        self.linear_block = nn.Sequential(
            nn.Linear(linear_input_shape, 1), 
            # nn.ELU(),
            # nn.Linear(50, 1)
        )
        self.linear_block.to(device)

    def forward(self, X):
        X = X.unsqueeze(1) 
        out = self.sequantial_block(X)
        out = self.linear_block(out)
        return out
        

            
        