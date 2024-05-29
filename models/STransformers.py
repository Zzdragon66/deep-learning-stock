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
class STransformer(GeneralModel):

    def __init__(self, embed_dim, nheads,
                 d_model : list, device):
        """Initialization of the Transformer 
        Args:
            input_dim (int): the input dimension D of X
            embed_dim (int): the embedding dimension 
            nheads (int): the number of heads 
            value_dim (int): the value dimension
        """
        super().__init__("Stransformer")
        # store the device
        self.device = device
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.multihead = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nheads,
            batch_first=True)
        # feedforward neural network
        # hard-coded the neural network
        forward_lst = OrderedDict()
        forward_lst["linear1"] = nn.Linear(embed_dim, d_model)
        forward_lst["act1"] = nn.Tanh()
        forward_lst["linear2"] = nn.Linear(d_model, d_model)
        forward_lst["act2"] = nn.Tanh()
        forward_lst["linear3"] = nn.Linear(d_model, embed_dim)
        forward_lst["act3"] = nn.Tanh()
        self.feedfoward = nn.Sequential(forward_lst)
        # norm may not be used 
        # TODO(Allen) : Check if we need t hte normalization
        # linear projection at the last layer 
        self.linear = nn.Linear(embed_dim, 1)
    
    def forward(self, X):
        # assume X has shape N, T, C
        N, T, C = X.shape

        X = self.pos_encoding(X) # positional encoding
        # generate the mask 
        mask = self.generate_square_subsequent_mask(T)
        mask = mask.to(self.device)
        
        # get the attention weight
        att_out, _ = self.multihead(X, X, X, attn_mask = mask) # N, T, val_dim
        # feedforward layer 
        feed_forward = self.feedfoward(att_out)
        # residual
        res_sum = feed_forward + att_out
        # final linear layer
        linear_proj = self.linear(res_sum)
        return linear_proj[:, -1, :]

    
    def generate_square_subsequent_mask(self, sz : int):
        """Generate the mask"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
