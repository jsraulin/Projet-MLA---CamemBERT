import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, AutoTokenizer # Ajout de AutoTokenizer
# from transformers import PreTrainedTokenizerFast # Commenté
from datasets import load_dataset
from itertools import chain

class CmbrtDataModule(pl.LightningDataModule):
    def __init__(self, 
                 train_files: list, 
                 tokenizer_path: str = "camembert-base", # Modifié par défaut
                 batch_size: int = 8, 
                 max_length: int = 512,
                 num_workers: int = 32):
        super().__init__()
        self.train_files = train_files
        self.tokenizer_path = tokenizer_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers

    def setup(self, stage=None):
        # --- ANCIEN TOKENIZER (COMMENTÉ) ---
        # self.tokenizer = PreTrainedTokenizerFast(
        #     tokenizer_file=self.tokenizer_path,
        #     model_max_length=self.max_length
        # )
        # self.tokenizer.bos_token = "<s>"
        # self.tokenizer.eos_token = "</s>"
        # self.tokenizer.unk_token = "<unk>"
        # self.tokenizer.pad_token = "<pad>"
        # self.tokenizer.mask_token = "<mask>"

        # --- NOUVEAU TOKENIZER (HUGGING FACE) ---
        # On utilise AutoTokenizer pour charger camembert-base proprement
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        # Chargement des données
        dataset = load_dataset("text", data_files={"train": self.train_files})

        # Tokenisation
        def tokenize_function(examples):
            # SentencePiece gère le texte brut directement
            return self.tokenizer(
                examples["text"], 
                truncation=True, 
                max_length=self.max_length, 
                return_special_tokens_mask=True
            )

        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            num_proc=self.num_workers
        )

        # Grouping pour remplir les séquences de 512
        def group_texts(examples):
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= self.max_length:
                total_length = (total_length // self.max_length) * self.max_length
            result = {
                k: [t[i : i + self.max_length] for i in range(0, total_length, self.max_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        self.train_dataset = tokenized_datasets["train"].map(
            group_texts,
            batched=True,
            num_proc=self.num_workers
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, 
                mlm=True, 
                mlm_probability=0.15
            ),
            pin_memory=True
        )