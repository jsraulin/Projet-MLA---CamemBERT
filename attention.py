# attention.py

" Étape 2 : Mécanisme d'attention multi-têtes. Chaque mot peut regarder les autres mots de la phrase."

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Projections linéaires pour Q, K, V
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _split_heads(self, x):
        # Sépare le vecteur caché en plusieurs têtes
        B, T, H = x.size()
        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        # Recolle toutes les têtes ensemble
        B, nh, T, hd = x.size()
        return x.permute(0, 2, 1, 3).contiguous().view(B, T, nh * hd)

    def forward(self, hidden_states, attention_mask=None):
        # Calcul des requêtes, clés, valeurs
        Q = self._split_heads(self.q_proj(hidden_states))
        K = self._split_heads(self.k_proj(hidden_states))
        V = self._split_heads(self.v_proj(hidden_states))

        # Produit scalaire entre Q et K (mesure de similarité)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Appliquer le masque (pour ignorer les tokens PAD)
        if attention_mask is not None:
            scores = scores + attention_mask

        # Poids d'attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Combiner les valeurs V selon les poids
        context = torch.matmul(attn_weights, V)
        context = self._merge_heads(context)

        # Projection finale
        output = self.out_proj(context)
        return output
