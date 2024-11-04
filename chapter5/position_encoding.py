import torch
import math
from torch import nn

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         # Create a constant positional encoding matrix with max_len x d_model
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
#         # Apply sin to even indices and cos to odd indices
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
        
#         # Register as a buffer (not a trainable parameter)
#         self.register_buffer('pe', pe.unsqueeze(0))

#     def forward(self, x):
#         # Add positional encoding to input embeddings
#         return x + self.pe[:, :x.size(1), :]
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))  # Shape (d_model/2)

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input tensor
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Example usage
pos_encoding = PositionalEncoding(d_model=512)
embeddings = torch.randn(32, 10, 512)  # batch_size=32, sequence_length=10, d_model=512
encoded = pos_encoding(embeddings)
print(pos_encoding)
print(embeddings.shape)
print(encoded.shape)
