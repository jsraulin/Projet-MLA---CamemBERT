from tokenizers import SentencePieceBPETokenizer
import glob

files = glob.glob("./DATASET_pour_camemBERT/*.txt")
vocab_size = 32000 

"""
cf. "We use a vocabulary size of 32k subword tokens. These subwords are learned 
on 10^7 sentences sampled randomly from the pretraining dataset.
"""
# Initialisation du tokenizer SentencePiece
tokenizer = SentencePieceBPETokenizer()

print(f"Entraînement du tokenizer sur {files}...")

# Entraînement
# min_frequency=2 : on ignore les tokens qui n'apparaissent qu'une seule fois (bruit)
# special_tokens obligatoires pour l'architecture RoBERTa/CamemBERT
tokenizer.train(
    files=files,
    vocab_size=vocab_size,
    min_frequency=2,
    show_progress=True,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

# Sauvegarde
# Cela va créer 2 fichiers : vocab.json et merges.txt (ou tokenizer.json)
tokenizer.save_model(".", "camembert-custom")
print("Tokenizer sauvegardé dans ./camembert-custom-vocab.json et merges.txt")