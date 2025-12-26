# test_model.py

"Petit test pour vérifier que le modèle tourne bien "


import torch
from config import CamembertConfig
from model import SimpleCamembertModel

if __name__ == "__main__":
    config = CamembertConfig()
    model = SimpleCamembertModel(config)

    # Exemple d'entrée : batch = 1 phrase de 6 tokens
    input_ids = torch.tensor([[5, 10, 20, 30, 2, 1]])
    attention_mask = (input_ids != 1).long()

    outputs = model(input_ids, attention_mask)
    print("Taille de sortie :", outputs.shape)  # (1, 6, vocab_size)
