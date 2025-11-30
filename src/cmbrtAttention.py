import torch
import torch.nn as nn

class CmbrtAttention(nn.Module):
    """
    Mécanisme d'auto-attention multi-têtes, calcule pour chaque token d'une séquence, 
    une représentation contextuelle en tenant compte de tous les autres tokens de la séquence.
    "BERT’s model architecture is a multi-layer bidirectional Transformer encoder based on the original 
    implementation described in Vaswani et al. (2017) and released in the tensor2tensor library.1 
    Because the use of Transformers has become common and our implementation is almost identical to the original"

    https://nlp.seas.harvard.edu/2018/04/03/attention.html#attention

    Alternative : encoder = nn.TransformerEncoder; encoder_layer = nn.TransformerEncoderLayer
    """
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        # Projection des hidden_states en Q, K, V
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # Séparation en têtes
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32))

        # Masquage (si attention_mask est fourni)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Softmax et Dropout
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # Pondération des valeurs
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer