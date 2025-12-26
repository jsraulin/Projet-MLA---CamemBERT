# cmbrt_lightning_module.py
# Entraînement Lightning pour notre CamemBERT simplifié
# Ici on fait du Masked Language Modeling (MLM) comme dans l'article CamemBERT

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from model import SimpleCamembertModel
from config import CamembertConfig


class CamembertLightning(pl.LightningModule):
    def __init__(self, vocab_size=32000, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()  # garde les arguments dans les logs
        config = CamembertConfig()
        config.vocab_size = vocab_size
        self.model = SimpleCamembertModel(config)
        self.lr = lr

    def forward(self, input_ids, attention_mask=None):
        """Passe avant (forward) du modèle"""
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        """Une étape d'entraînement"""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        outputs = self(input_ids, attention_mask)

        # Calcul de la perte uniquement sur les tokens masqués
        loss = F.cross_entropy(outputs.view(-1, self.hparams.vocab_size),
                               labels.view(-1),
                               ignore_index=-100)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Optimiseur : AdamW, comme dans BERT/CamemBERT"""
        return AdamW(self.parameters(), lr=self.lr)
