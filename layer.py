import torch.nn as nn
from attention import MultiHeadSelfAttention

class TransformerEncoderLayer(nn.Module):
    """Une couche Transformer : attention + feed-forward."""
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        attn_out = self.attention(hidden_states, attention_mask)
        hidden_states = self.layer_norm1(hidden_states + self.dropout(attn_out))

        ff_out = self.feed_forward(hidden_states)
        hidden_states = self.layer_norm2(hidden_states + self.dropout(ff_out))
        return hidden_states
