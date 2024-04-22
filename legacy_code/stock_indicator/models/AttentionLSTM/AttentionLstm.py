import torch
import torch.nn as nn 


class AttentionLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.mulihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, bias=False
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        out, _ = self.lstm(X)
        # X has shape of N, T, D
        # Attention has input N, L, E=D

        N, seq_len, _ = out.shape

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(out.device)  # Move mask to the right device

        attn_output, attn_output_weights = self.mulihead_attention(out, out, out, attn_mask=mask)
        attn_output = self.attention_norm(attn_output)
        return self.fc(attn_output[:, -1, :])
    

    def forward_with_attention(self, X):
        out, _ = self.lstm(X)
        # X has shape of N, T, D
        # Attention has input N, L, E=D

        N, seq_len, _ = out.shape

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(out.device)  # Move mask to the right device

        attn_output, attn_output_weights = self.mulihead_attention(out, out, out, attn_mask=mask)
        attn_output = self.attention_norm(attn_output)
        print(attn_output.shape)
        return self.fc(attn_output[:, -1, :]), attn_output_weights