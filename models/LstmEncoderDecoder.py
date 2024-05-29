from GeneralModel import GeneralModel
import torch 
import torch.nn as nn
from collections import OrderedDict 

# Assume the input has the format N, T, C
# Assume the dummy input is on GPU

# The convolutional block for the encoder
class ConvBlock(nn.Module):
    def __init__(self, kernel_size, max_pool_size, in_channel, out_channel):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=0),
            nn.MaxPool2d(kernel_size=max_pool_size)
        )
    
    def forward(self, X):
        return self.convblock(X)


class CLstmEncoder(GeneralModel):
    """The CLSTM encoder for the stock data"""

    def __init__(self, dummy_input, 
                 kernel_sizes=[(1, 1)],
                 max_pool_sizes=[(1, 1)],
                 filter_sizes=[15],
                 hidden_size=50, 
                 model_name="CLstmEncoder",
                 num_layers=1):
        super().__init__(model_name)
        # Start of the CnnBlock
        N, T, C = dummy_input.shape 
        cnn_dummy_input = dummy_input.unsqueeze(-1).swapaxes(1, 2)
        sequential_block = OrderedDict()
        for i, (kernel_size, max_pool_size, filter_size) in enumerate(zip(kernel_sizes, max_pool_sizes, filter_sizes)):
            if i == 0:
                sequential_block[f'ConvBlock-{i}'] = ConvBlock(
                    kernel_size, max_pool_size, in_channel=C, out_channel=filter_size
                )
            else:
                sequential_block[f'ConvBlock-{i}'] = ConvBlock(
                    kernel_size, max_pool_size, in_channel=filter_sizes[i - 1], out_channel=filter_size
                )
        self.sequential_block = nn.Sequential(sequential_block)
        self.sequential_block.to(dummy_input.device)
        N, C_out, T, _ = self.sequential_block(cnn_dummy_input).shape
        # Start of Lstm Encoder
        self.lstm = nn.LSTM(
            input_size=C_out, hidden_size=hidden_size, num_layers=num_layers, 
            batch_first=True
        )
        # Store the cout so that we can have access to the data
        self.cout = C_out
    
    def get_cout(self):
        return self.cout
    
    def forward(self, X):
        """Forward method"""
        # X shape: N, T, C
        reshaped_X = X.swapaxes(1, 2).unsqueeze(-1)
        out = self.sequential_block(reshaped_X)
        out = out.swapaxes(1, 2)
        out = out.squeeze(-1)
        out, (hidden, cell) = self.lstm(out)
        return hidden, cell


class LstmDecoder(GeneralModel):
    """The Lstm decoder"""
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__("LstmDecoder")
        
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.linear(output)
        return prediction, hidden, cell


class LstmEncoderDecoder(GeneralModel):

    def __init__(self, encoder, decoder, out_len=5, out_size=1):
        super().__init__("LstmEncoderDecoder")
        self.encoder = encoder
        self.decoder = decoder
        self.out_len = out_len
        self.out_size = out_size
    

    def forward(self, X):
        N, T, C = X.shape
        outputs = torch.zeros(N, self.out_len, device=X.device)

        hidden, cell = self.encoder(X)

        # Start token
        inputs = X[:, -1, -2].unsqueeze(-1).unsqueeze(-1)
        for t in range(self.out_len):
            prediction, hidden, cell = self.decoder(inputs, hidden, cell)
            outputs[:, t] = prediction.squeeze(-1).squeeze(-1)
            inputs = prediction
        return outputs
