# embeddings.py

" Étape 1 : Transformer les tokens (entiers) en vecteurs. On ajoute aussi la position des mots dans la phrase."


import torch
import torch.nn as nn

class CamembertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Embedding des mots
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        # Embedding de position (ordre des mots)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        # Normalisation + dropout pour régularisation
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        # input_ids : (batch_size, longueur_phrase)
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # On crée les positions [0, 1, 2, 3, ...]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # On additionne les embeddings mots + positions
        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)

        # Normalisation et régularisation
        x = self.layer_norm(x)
        x = self.dropout(x)

        return x  # (batch, longueur, dimension)
