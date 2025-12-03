import torch
import torch.nn as nn

class CamembertEmbeddings(nn.Module):
    """Convertit les tokens en vecteurs (embedding + position)."""
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x  # (B, T, H)
