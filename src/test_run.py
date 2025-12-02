import torch
import torch.nn as nn
from transformers import CamembertConfig
from camembert_proj import CmbrtModel

# Charger la configuration
config = CamembertConfig.from_pretrained("camembert-base")

# Instancier le modèle
model = CmbrtModel(config)

# Exemple d'entrée
input_ids = torch.randint(0, config.vocab_size, (2, 10))  # batch_size=2, seq_len=10

# Forward pass
logits = model(input_ids)
print("Vocab size : ", config.vocab_size)
print(logits.shape)  # Doit afficher torch.Size([2, 10, vocab_size])
