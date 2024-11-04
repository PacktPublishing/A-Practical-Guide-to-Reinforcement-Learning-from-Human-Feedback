import torch
import torch.nn as nn
from position_encoding import PositionalEncoding

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        # Split the embedding into multiple heads
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, depth)

    def forward(self, q, k, v, mask):
        batch_size = q.size(0)

        # Linear projections
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled dot-product attention
        scaled_attention, _ = self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)

        return self.dense(concat_attention)

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)
        dk = k.size(-1)
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = nn.functional.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len, depth)

        return output, attention_weights

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        attn_output = self.attention(x, x, x, mask)
        out1 = self.layernorm1(x + self.dropout1(attn_output))  # Residual connection
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output))  # Residual connection

def test_transformer_components():
    # Testing positional encoding
    pos_encoding = PositionalEncoding(d_model=64)
    sample_input = torch.randn(10, 50, 64)  # (batch_size, seq_len, d_model)
    pos_encoded_output = pos_encoding(sample_input)
    assert pos_encoded_output.shape == sample_input.shape, "Positional encoding test failed!"

    # Testing multi-head attention
    self_attn = MultiHeadAttention(d_model=64, num_heads=8)
    attn_output = self_attn(sample_input, sample_input, sample_input, mask=None)
    assert attn_output.shape == sample_input.shape, "Multi-head attention test failed!"

    # Testing encoder layer
    enc_layer = TransformerEncoderLayer(d_model=64, num_heads=8, d_ff=256)
    encoder_output = enc_layer(sample_input, None)
    assert encoder_output.shape == sample_input.shape, "Encoder layer test failed!"

    print("All tests passed!")

# Run the test
test_transformer_components()
