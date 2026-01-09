# config.py
# Charge la configuration officielle de CamemBERT depuis Hugging Face

from transformers import CamembertConfig

def get_config():
    """La configuration officielle de CamemBERT"""
    return CamembertConfig.from_pretrained("camembert-base")

if __name__ == "__main__":
    print(get_config())
