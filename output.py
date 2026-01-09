# output.py

"Étape 4 : Tête de sortie pour le Masked Language Modeling (MLM). Transforme les vecteurs cachés en probabilités sur le vocabulaire."


import torch
import torch.nn as nn

class CamembertOutputHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states)
        return logits
