import torch
import torch.nn as nn
import math
from collections import OrderedDict
from GeneralModel import GeneralModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (N, T, C)
        x = x + self.pe[:, :x.size(1), :]  # Broadcasting across the batch dimension
        return x 

class ConvBlock(nn.Module):
    def __init__(self, kernel_size, max_pool_size, in_channel, out_channel):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=0),
            nn.MaxPool2d(kernel_size=max_pool_size)
        )
    def forward(self, X):
        return self.convblock(X)

    
class SCTransformerEncoder(GeneralModel):

    def __init__(self, dummy_input,
                 embed_dim, 
                 nheads,
                 d_model : list, 
                 device,
                 kernel_size = (1, 1),
                 max_pool_size = (1, 1),
                 filter_size = 28,
                 ):
        """Initialization of the Transformer 
        Args:
            input_dim (int): the input dimension D of X
            embed_dim (int): the embedding dimension 
            nheads (int): the number of heads 
            value_dim (int): the value dimension
        """
        super().__init__("SCtransformerEncoder")
        # store the device
        self.device = device
        # Use the convolution layer first
        N, T, C = dummy_input.shape 
        cnn_dummy_input = dummy_input.unsqueeze(-1).swapaxes(1, 2)
        self.convlayer = ConvBlock(kernel_size, max_pool_size, embed_dim, out_channel=filter_size)
        self.convlayer.to(device)
        N, C_out, T, _ = self.convlayer(cnn_dummy_input).shape
        self.cout = C_out
        # End
        self.pos_encoding = PositionalEncoding(C_out)
        self.multihead = nn.MultiheadAttention(
            embed_dim=C_out,
            num_heads=nheads,
            batch_first=True)
        # feedforward neural network
        # hard-coded the neural network
        forward_lst = OrderedDict()
        forward_lst["linear1"] = nn.Linear(C_out, d_model)
        forward_lst["act1"] = nn.Tanh()
        forward_lst["linear2"] = nn.Linear(d_model, d_model)
        forward_lst["act2"] = nn.Tanh()
        forward_lst["linear3"] = nn.Linear(d_model, C_out)
        forward_lst["act3"] = nn.Tanh()
        self.feedfoward = nn.Sequential(forward_lst)
        # use the lstm to produce the hidden and cell state for the lstm decoder
        self.lstm = nn.LSTM(input_size = C_out, 
                            hidden_size= 2 * C_out, 
                            batch_first=True)

        
    
    def forward(self, X):
        # assume X has shape N, T, C
        N, T, C = X.shape
        # ConvBlock first 
        reshaped_X = X.swapaxes(1, 2).unsqueeze(-1)
        reshaped_X = self.convlayer(reshaped_X)
        X = reshaped_X.swapaxes(1, 2).squeeze(-1) # N, T, filter_size
        # Pos Encoding
        X = self.pos_encoding(X)
        # generate the mask 
        mask = self.generate_square_subsequent_mask(T)
        mask = mask.to(self.device)
        
        # get the attention weight
        att_out, _ = self.multihead(X, X, X, attn_mask = mask) # N, T, val_dim
        # feedforward layer 
        feed_forward = self.feedfoward(att_out)
        # residual
        res_sum = feed_forward + att_out
        pred, (hidden, cell) = self.lstm(res_sum)
        return hidden, cell
    
    def get_cout(self):
        return self.cout

    def generate_square_subsequent_mask(self, sz : int):
        """Generate the mask"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class LstmDecoder(GeneralModel):
    """LstmDecoder"""

    def __init__(self, input_dim, cout, device):
        super().__init__("LstmDecoder")
        self.device = device
        self.lstm = nn.LSTM(input_size = input_dim,
                            hidden_size = 2 * cout,
                            batch_first = True)
        self.linear = nn.Linear(2 * cout, 1)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.linear(output)
        return prediction, hidden, cell
        

    
class SCTransformerSeq(GeneralModel):
    """ScTransformer Seq to Seq"""

    def __init__(self, encoder, decoder, out_len = 5):
        super().__init__("SCTransformerSeq")
        self.encoder = encoder
        self.decoder = decoder
        self.out_len = out_len

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
        




