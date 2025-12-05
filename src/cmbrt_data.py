import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import CamembertTokenizer

#TODO: create DataSet class that tokenizes the data (fr_part1, 2, 3), takes (train_data, tokenizer).

class CmbrtDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base") # TODO: change, make own tokenizer based on SentencePiece

    def setup(self, stage=None):
        # TODO: self.train_dataset = DataSet(data, self.tokenizer) classused to train
        pass

    def train_dataloader(self):
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        pass