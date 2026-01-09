import torch
from transformers import CamembertTokenizer
from cmbrt_lightning_module import CamembertLightning

# 1. Charger le modèle entraîné 
checkpoint_path = "checkpoints/camembert-epoch=02-train_loss=3.034.ckpt/"
model = CamembertLightning.load_from_checkpoint(checkpoint_path)
model.eval()
#model.cuda()

# 2. Charger le tokenizer CamemBERT officiel 
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

# 3. Fonction de prédiction 
def predict_masked_token(sentence, top_k=5):
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"]#.cuda()
    attention_mask = inputs["attention_mask"]#.cuda()

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        vocab_size = model.model.config.vocab_size
        logits = outputs.view(-1, vocab_size)

    mask_token_index = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()
    probs = torch.softmax(logits[mask_token_index], dim=0)
    top_k_ids = torch.topk(probs, top_k).indices.tolist()

    print(f"\nPhrase : {sentence}")
    print(f"Top {top_k} prédictions pour {tokenizer.mask_token} :")
    for idx in top_k_ids:
        print(f"  - {tokenizer.decode([idx])}")

#  4. Tests 
phrases = [
    f"Le chat {tokenizer.mask_token} sur le canapé.",
    f"La voiture {tokenizer.mask_token} sur la route.",
    f"Le président {tokenizer.mask_token} un discours.",
    f"Il fait très {tokenizer.mask_token} aujourd'hui.",
    f"Les enfants {tokenizer.mask_token} à l'école."
]

for s in phrases:
    predict_masked_token(s)
    print("-" * 50)
