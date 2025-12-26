# model.py

" Étape 5 : Assemblage complet du modèle CamemBERT simplifié. Embeddings → plusieurs couches Transformer → tête de sortie. "

import torch.nn as nn
from embeddings import CamembertEmbeddings
from layer import TransformerEncoderLayer
from output import CamembertOutputHead

class SimpleCamembertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = CamembertEmbeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.output_head = CamembertOutputHead(config)

        # Poids partagés entre embeddings et sortie (technique BERT)
        self.output_head.decoder.weight = self.embeddings.word_embeddings.weight

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embeddings(input_ids)

        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        logits = self.output_head(hidden_states)
        return logits
