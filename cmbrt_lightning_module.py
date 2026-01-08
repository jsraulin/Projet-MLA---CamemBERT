import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from model import SimpleCamembertModel
from config import get_config


class CamembertLightning(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        config = get_config()
        self.model = SimpleCamembertModel(config)
        self.config = config
        self.lr = lr

        print(f"[INFO] CamemBERT config chargée : vocab_size = {config.vocab_size}")

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = (input_ids != self.config.pad_token_id).long()
        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(outputs.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=-100)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = (input_ids != self.config.pad_token_id).long()
        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(outputs.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=-100)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
