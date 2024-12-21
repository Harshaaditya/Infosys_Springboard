import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torch.cuda.amp import GradScaler, autocast
from datasets import load_dataset

# Define the checkpoint directory
checkpoint_dir = "fine_tuned_gpt2"

# Check if the model checkpoint exists
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

print("Training a new model")

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Setup the model for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load and tokenize the dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
dataset = dataset.select(range(1000))  # Limiting to the first 1000 samples

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids"])

# DataLoader for training
train_dataloader = DataLoader(tokenized_dataset, batch_size=16, shuffle=True)

# Optimizer and Mixed Precision Setup
optimizer = AdamW(model.parameters(), lr=5e-5)
scaler = GradScaler()
epochs = 2

# Training Loop
for epoch in range(epochs):
    model.train()
    loop = tqdm(train_dataloader, leave=True)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)

        # Forward pass with mixed precision
        with autocast():
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss

        # Backward pass with mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item())

    # Save the model and tokenizer after each epoch
    try:
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Model saved after epoch {epoch + 1} to {checkpoint_dir}")
    except Exception as e:
        print(f"Error saving model: {e}")

print(f"Model training completed after {epochs} epochs.")