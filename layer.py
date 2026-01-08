# layer.py
# Étape 3 : Définition d'une couche Transformer (Encoder Layer)

# Chaque couche combine :
#   Une attention multi-têtes (self-attention)
#   Une connexion résiduelle + normalisation
#   Un réseau feed-forward
#   Une autre connexion résiduelle + normalisation

import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadSelfAttention


class TransformerEncoderLayer(nn.Module):
    """
    Couche Transformer :

    - Gère les dépendances contextuelles via la self-attention.
    - Applique un réseau feed-forward pour affiner les représentations.
    - Intègre des connexions résiduelles et des normalisations (LayerNorm)
      pour stabiliser l’apprentissage et éviter l’explosion du gradient.
    """
    def __init__(self, config):
        super().__init__()

        # 1. Bloc d'attention multi-têtes
        self.self_attention = MultiHeadSelfAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 2. Bloc feed-forward
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

        # 3. Normalisation et régularisation
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: tenseur d’entrée de taille (batch_size, seq_len, hidden_size)
            attention_mask: masque optionnel pour ignorer les tokens [PAD]
        Returns:
            hidden_states mis à jour après attention + feed-forward
        """
        # == Bloc 1 : Multi-Head Self-Attention ==
        # Chaque token regarde tous les autres dans la séquence
        attn_output = self.self_attention(hidden_states, attention_mask)

        # Connexion résiduelle + normalisation
        hidden_states = self.layer_norm1(hidden_states + self.dropout(attn_output))

        # == Bloc 2 : Réseau Feed-Forward ==
        # Deux couches linéaires séparées par l’activation GELU
        ff_output = self.fc2(F.gelu(self.fc1(hidden_states)))

        # Connexion résiduelle + normalisation
        hidden_states = self.layer_norm2(hidden_states + self.dropout(ff_output))

        return hidden_states
