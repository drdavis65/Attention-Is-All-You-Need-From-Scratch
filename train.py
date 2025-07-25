import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from model import Transformer

# Hyperparameters
d_model = 512
context_window = 5000
warmup_steps = 4000
batch_size = 32
total_steps = 10000 

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")
vocab_size = tokenizer.vocab_size
pad_id = tokenizer.pad_token_id

# Load dataset
dataset = load_dataset("wmt14", "fr-en")


# Tokenization
def tokenize(example):
    en_texts = [x["en"] for x in example["translation"]]
    fr_texts = [x["fr"] for x in example["translation"]]
    return tokenizer(
        en_texts,
        text_target=fr_texts,
        max_length=context_window,
        padding="max_length",
        truncation=True
    )

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])

train_dataset = tokenized_dataset["train"]
valid_dataset = tokenized_dataset["validation"]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(vocab_size=vocab_size, d_model=d_model).to(device)

optimizer = Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)

def get_lr_schedule(optimizer, d_model, warmup_steps):
    def lr_lambda(step):
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps**-1.5)
    return LambdaLR(optimizer, lr_lambda)

scheduler = get_lr_schedule(optimizer, d_model, warmup_steps)
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

model.train()
global_step = 0
steps_per_epoch = len(train_dataset) // batch_size
num_epochs = total_steps // steps_per_epoch

best_loss = float('inf')
train_losses = []
lrs = []
val_losses = []

for epoch in range(num_epochs):
    running_loss = 0.0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["labels"].to(device)

        decoder_input = target_ids[:, :-1]
        decoder_target = target_ids[:, 1:]

        logits = model(input_ids, decoder_input)
        B, Tm1, V = logits.shape
        loss = loss_fn(logits.reshape(-1, V), decoder_target.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.detach().item()
        global_step += 1

        if global_step >= total_steps:
            break

    avg_loss = running_loss / len(train_loader)
    lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | LR: {lr:.6e}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(model.state_dict(), best_path)
        print(f"Saved best model with loss {best_loss:.4f}")

    if (epoch + 1) % 2 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"transformer_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    train_losses.append(avg_loss)
    lrs.append(lr)

    if global_step >= total_steps:
        print("Reached total training steps.")
        break

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["labels"].to(device)

            decoder_input = target_ids[:, :-1]
            decoder_target = target_ids[:, 1:]

            logits = model(input_ids, decoder_input)
            B, Tm1, V = logits.shape
            loss = loss_fn(logits.reshape(-1, V), decoder_target.reshape(-1))
            val_loss += loss.item()

    val_loss /= len(valid_loader)
    print(f"Validation Loss: {val_loss:.4f}")
    val_losses.append(val_loss)
    model.train()

print("Training complete.")

plt.figure(figsize=(10, 4))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.grid(True)
plt.legend()

# LR Plot
plt.subplot(1, 2, 2)
plt.plot(lrs, label="Learning Rate", color="orange")
plt.xlabel("Epoch")
plt.ylabel("LR")
plt.title("Learning Rate Schedule")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, "training_progress.png"))
plt.show()
