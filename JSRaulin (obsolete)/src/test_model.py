import torch
# On utilise PreTrainedTokenizerFast pour les fichiers .json
from transformers import PreTrainedTokenizerFast 
from cmbrt_module import CmbrtLightningModule

# --- CONFIGURATION ---
# Chemin vers votre checkpoint (.ckpt)
CHECKPOINT_PATH = "/Users/jsraulin/Desktop/MLA - Proj/Projet-MLA---CamemBERT/src/checkpoints/epoch=3-step=34432.ckpt"

# Chemin EXACT vers votre fichier tokenizer.json
# Exemple : "/Users/jsraulin/Desktop/mon_projet/tokenizer.json"
TOKENIZER_FILE = "/Users/jsraulin/Desktop/MLA - Proj/Projet-MLA---CamemBERT/src/tokenizer.json" 

def load_model():
    print(f"Chargement du modèle depuis : {CHECKPOINT_PATH}...")
    
    # 1. Configuration du modèle (Forcez la taille 6 layers pour le test)
    # ATTENTION : Vérifiez que vocab_size correspond bien à la taille de votre tokenizer JSON !
    # Si votre json a 30000 tokens et le modèle 32005, mettez 32005 ici (le modèle a du padding).
    model = CmbrtLightningModule(
        vocab_size=32000, 
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        learning_rate=1e-4
    )
    
    # 2. Chargement des poids
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
    load_result = model.load_state_dict(checkpoint["state_dict"], strict=False)
    print(f"Poids chargés. Info: {load_result}")
    
    model.eval()
    model.freeze()
    
    # 3. Chargement du Tokenizer JSON
    print(f"Chargement du tokenizer depuis : {TOKENIZER_FILE}...")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=TOKENIZER_FILE,
        # Il est CRUCIAL de redéfinir les tokens spéciaux ici pour que le masque fonctionne
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        sep_token="</s>",
        pad_token="<pad>",
        cls_token="<s>",
        mask_token="<mask>"
    )
    
    print(f"Taille du vocabulaire tokenizer : {len(tokenizer)}")
    return model, tokenizer

def predict_mask(model, tokenizer, text, top_k=5):
    if "<mask>" not in text:
        print(f"⚠️ Erreur : La phrase doit contenir '<mask>'.")
        return

    # Préparation de l'entrée
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        output = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = output.logits if hasattr(output, 'logits') else output

    # Trouver l'index du token <mask>
    try:
        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    except IndexError:
        print("❌ Erreur: Le tokenizer n'a pas reconnu le token <mask> dans la phrase.")
        return

    mask_logits = logits[0, mask_token_index, :]
    top_tokens = torch.topk(mask_logits, top_k, dim=1).indices[0].tolist()

    print(f"\nPhrase : {text}")
    print("Prédictions :")
    for token_id in top_tokens:
        word = tokenizer.decode([token_id])
        print(f"  - {word}")

# --- EXECUTION ---
if __name__ == "__main__":
    try:
        model, tokenizer = load_model()
        
        predict_mask(model, tokenizer, "Le chat mange la <mask> .")
        predict_mask(model, tokenizer, "Paris est en <mask> .")
        
    except Exception as e:
        print(f"❌ Une erreur est survenue : {e}")