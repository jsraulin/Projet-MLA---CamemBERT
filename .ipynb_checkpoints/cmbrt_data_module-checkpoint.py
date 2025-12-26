# cmbrt_data_module.py
# Gestion des données et masquage automatique pour le MLM
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CamembertTokenizer, DataCollatorForLanguageModeling


class CamembertDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, batch_size=8, max_length=128):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.max_length = max_length

    def setup(self, stage=None):
        """Prépare le tokenizer et le dataset"""
        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

        dataset = load_dataset("text", data_files={"train": self.dataset_path})

        def tokenize_function(examples):
            return self.tokenizer(examples["text"],
                                  truncation=True,
                                  padding="max_length",
                                  max_length=self.max_length)

        self.dataset = dataset.map(tokenize_function, batched=True)
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator
        )
