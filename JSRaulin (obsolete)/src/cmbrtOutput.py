import torch.nn as nn
import torch

class CmbrtOutputHead(nn.Module):
    """
    Tête de sortie pour le Masked Language Modeling (MLM).
    Projette les hidden_states vers le vocabulaire et applique un softmax (via la loss).
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))  # Initialisation du biais

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states) + self.bias  # Ajout du biais
        return logits  # Pas de softmax ici (appliqué dans la loss)