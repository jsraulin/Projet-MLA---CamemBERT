import numpy
import torch
import torch.nn
from transformers import CamembertConfig

cfg = CamembertConfig.from_pretrained("camembert-base") #

gelu = torch.nn.GELU()

class CmbrtModel():
    """
    Uses the original BERT_base archictecture, we use the available config file
    for parameter retrieval.
    BERT's own model "architecture is a multi-layer bidirectional Transformer encoder
    based on the original implementation described in Vaswani et al. (Attention is 
    all you need, 2017) and released in the tensor2tensor library" (cf. tensorflow).

    Model's general architecture includes:
        - Word embeddings TODO: make embedding class (use nn.Embeddings?, cf. BERT
          for special tokens; + pos embeddings)
        - Individual Transformer layer (cf. transformer encoder arch.), TODO: all 
            ->Embeddings->Multi-head self attention->Add&Norm->Feed Forward->Add&Norm->Linear->Softmax->Output
        - Encoder TODO: make encoder class (this is where the transformer layers go, 
        forward: collect outputs from each layers) 
    """
    def __init__(self, config):
        pass