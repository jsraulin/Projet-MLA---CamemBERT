# config.py
"Ce fichier charge directement la configuration officielle de CamemBERT depuis Hugging Face"

from transformers import CamembertConfig

config = CamembertConfig.from_pretrained("camembert-base")



if __name__ == "__main__":
    print(config)
