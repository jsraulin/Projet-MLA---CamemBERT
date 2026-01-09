import torch.nn as nn
from cmbrtAttention import CmbrtAttention

class CmbrtLayer(nn.Module):
    """
    Couche Transformer (encoder only): Tête d'attention, puis un réseau feed-forward.
    """
    def __init__(self, config):
        super().__init__()
        self.attention = CmbrtAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        # Self-Attention
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = hidden_states + attention_output
        hidden_states = self.layer_norm1(hidden_states)

        # Feed Forward
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = hidden_states + feed_forward_output
        hidden_states = self.layer_norm2(hidden_states)

        return hidden_states