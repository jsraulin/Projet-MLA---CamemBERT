import numpy
import torch
import torch.nn as nn
from transformers import CamembertConfig

from cmbrtEmbeddings import CmbrtEmbeddings
from cmbrtEncoder import CmbrtEncoder
from cmbrtOutput import CmbrtOutputHead
cfg = CamembertConfig.from_pretrained("camembert-base")

gelu = torch.nn.GELU()

class CmbrtModel(nn.Module):
    """
    Uses the original BERT_base archictecture, we use the available config file
    for parameter retrieval.
    BERT's own model "architecture is a multi-layer bidirectional Transformer encoder
    based on the original implementation described in Vaswani et al. (Attention is 
    all you need, 2017) and released in the tensor2tensor library" (cf. tensorflow).

    Model's general architecture includes:
        - Word embeddings: make embedding class (use nn.Embeddings?, cf. BERT
          for special tokens; + pos embeddings)
        - Individual Transformer layer (cf. transformer encoder arch.): 
            ->Embeddings(->Multi-head self attention->Add&Norm->Feed Forward->Add&Norm->Linear->Softmax->Output)
        - Encoder: encoder class (this is where the transformer layers go, 
        forward: collect outputs from each layers) 
    """
    def __init__(self, config):
        super().__init__()
        self.embeddings = CmbrtEmbeddings(config)
        self.encoder = CmbrtEncoder(config)
        self.output_head = CmbrtOutputHead(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # 1. Embeddings
        embeddings = self.embeddings(input_ids, token_type_ids)
        # 2. Encoder
        encoder_output = self.encoder(embeddings, attention_mask)
        # 3. Tête de sortie
        logits = self.output_head(encoder_output)
        return logits