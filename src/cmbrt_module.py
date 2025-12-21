import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import CamembertConfig
from camembert_proj import CmbrtModel

class CmbrtLightningModule(pl.LightningModule):
    def __init__(self, 
                 vocab_size: int = 32005,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 max_position_embeddings: int = 512,
                 learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters() # Sauvegarde la config pour les logs
        self.config_obj = CamembertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=1
        )   
        self.model = CmbrtModel(self.config_obj)
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        return self.model(input_ids, attention_mask, token_type_ids)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # On gère le masque dans le CmbrtModel
        logits = self(input_ids, attention_mask)
        
        # Reshape pour la Loss (Batch * Seq_len, Vocab_Size)
        loss = self.loss_fn(logits.view(-1, self.hparams.vocab_size), labels.view(-1))  
              
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.98)) # cf. Optimisation (p5) -> β