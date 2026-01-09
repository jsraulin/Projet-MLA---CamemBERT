import torch
import torch.nn as nn
from transformers import CamembertConfig

cfg = CamembertConfig.from_pretrained("camembert-base")

class CmbrtEmbeddings(nn.Module):
    """
    Module d'entrée de CmbrtModel.
    "The input embeddings are the sum of the token embeddings, the segmentation embeddings and the position embeddings"
    (BERT, partie 3).
    Dans RoBERTa (et donc dans CamemBERT), les embeddings de segmentation (token_type_embeddings) ne sont pas utilisés 
    (toujours a 0).

    """
    def __init__(self, config):
        super().__init__()
        # Embeddings de token
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        # Embeddings de position
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # Embeddings de segment (toujours zéros pour CamemBERT)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # LayerNorm et Dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        # Taille du batch et de la séquence
        seq_length = input_ids.size(1)
        # Générer les positions (0 à seq_length-1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        # Si token_type_ids n'est pas fourni, utiliser des zéros (comme dans CamemBERT)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Récupérer les embeddings
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # Somme des embeddings
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        # LayerNorm et Dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings