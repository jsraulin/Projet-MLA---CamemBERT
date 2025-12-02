import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import CamembertTokenizer

def tokenize_function(examples, tokenizer=None):
    if not tokenizer:
        tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")

def train(model, dataloader, optimizer, device, epochs=3):
    model.train()
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

#TODO evaluations, DataLoader (not here), save/load, test training with optimizer