import torch
from config import CamembertConfig
from model import SimpleCamembertModel

# Création du modèle
config = CamembertConfig()
model = SimpleCamembertModel(config)

# Exemple : une phrase avec 4 tokens
input_ids = torch.tensor([[5, 10, 20, 2]])

# Passage dans le modèle
outputs = model(input_ids)

print("Taille de sortie :", outputs.shape)
print("Extrait du vecteur du 1er token :", outputs[0, 0, :10])
