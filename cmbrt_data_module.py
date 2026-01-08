"""
CamemBERT DataModule optimisé :
- Entraîne sur 20 % de fr_part_1.txt + fr_part_2.txt
- Valide sur 2 % de fr_part_3.txt
- Tokenisation sauvegardée (pour ne pas la refaire)
"""

import os
from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import CamembertTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class CamembertDataModule(pl.LightningDataModule):
    def __init__(self, base_path="/home/camembert/dataset_g5", batch_size=16, max_length=128, num_workers=8):
        super().__init__()
        self.base_path = base_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers

        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

    def setup(self, stage=None):
        os.makedirs("data_tokenized", exist_ok=True)

        # Charger si déjà tokenisé
        if os.path.exists("data_tokenized/train") and os.path.exists("data_tokenized/val"):
            print("[INFO] Chargement des datasets tokenisés depuis le disque ")
            self.train_dataset = load_from_disk("data_tokenized/train")
            self.val_dataset = load_from_disk("data_tokenized/val")
            print(f"[INFO] Train : {len(self.train_dataset):,} / Val : {len(self.val_dataset):,}")
            return

        print("\n[INFO] Chargement des fichiers texte...")
        train_files = {
            "part1": os.path.join(self.base_path, "fr_part_1.txt"),
            "part2": os.path.join(self.base_path, "fr_part_2.txt"),
        }
        val_file = os.path.join(self.base_path, "fr_part_3.txt")

        raw_train = load_dataset("text", data_files=train_files)
        raw_val = load_dataset("text", data_files={"validation": val_file})

        dataset_train = concatenate_datasets([raw_train["part1"], raw_train["part2"]])
        dataset_val = raw_val["validation"]

        print(f"[INFO] Total train : {len(dataset_train):,} lignes")
        print(f"[INFO] Total val (avant réduction) : {len(dataset_val):,} lignes")

        # --- Réduction 
        train_ratio = 0.5
        val_ratio = 0.1

        train_subset_size = int(len(dataset_train) * train_ratio)
        val_subset_size = int(len(dataset_val) * val_ratio)

        dataset_train = dataset_train.select(range(train_subset_size))
        dataset_val = dataset_val.select(range(val_subset_size))

        print(f"[INFO] Train réduit à {len(dataset_train):,} lignes (50 %)")
        print(f"[INFO] Validation réduite à {len(dataset_val):,} lignes (10 %) ")

        # --- Tokenisation ---
        print("[INFO] Tokenisation en cours (première fois seulement)...")

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )

        tokenized_train = dataset_train.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        tokenized_val = dataset_val.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        print("[INFO] Tokenisation terminée ")

        # Sauvegarde
        print("[INFO] Sauvegarde des datasets tokenisés...")
        tokenized_train.save_to_disk("data_tokenized/train")
        tokenized_val.save_to_disk("data_tokenized/val")
        print("[INFO] Sauvegarde terminée ")

        self.train_dataset = tokenized_train
        self.val_dataset = tokenized_val

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.data_collator,
        )


if __name__ == "__main__":
    dm = CamembertDataModule(
        base_path="/home/camembert/dataset_g5",
        batch_size=16,
        max_length=128,
        num_workers=8,
    )
    dm.setup()
    print("\n[INFO] Vérification d’un batch :")
    batch = next(iter(dm.train_dataloader()))
    print("input_ids shape :", batch["input_ids"].shape)
    print("labels shape    :", batch["labels"].shape)
