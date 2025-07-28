import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import MarianTokenizer

import matplotlib.pyplot as plt

from model import Transformer

# Hyperparameters
d_model = 512
context_window = 5000
warmup_steps = 4000
batch_size = 32
total_steps = 10000 

# Tokenizer
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
vocab_size = tokenizer.vocab_size
pad_id     = tokenizer.pad_token_id

# Dataset english to french
dataset = load_dataset("wmt14", "fr-en")
dataset["train"] = dataset["train"].shuffle(seed=42)


# === Setup ===
chunk_size = 400_000
total_train_size = len(dataset["train"])
num_rounds = total_train_size // chunk_size
print(f"num rounds: {num_rounds}", flush=True)
round = 0
start_idx = round * chunk_size
end_idx = min(start_idx + chunk_size, total_train_size)
print(f"\nStarting Round {round + 1}: Training on examples [{start_idx} : {end_idx}]", flush=True)


checkpoint_dir = "/content/drive/MyDrive/transformer_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(vocab_size=vocab_size, d_model=d_model).to(device)

base_lr = d_model ** -0.5
optimizer = Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.98), eps=1e-9)

def get_lr_schedule(optimizer, warmup_steps):
    def lr_lambda(step):
        step = max(step, 1)
        return min(step ** -0.5, step * warmup_steps ** -1.5)
    return LambdaLR(optimizer, lr_lambda)


scheduler = get_lr_schedule(optimizer, warmup_steps)
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)

global_step = 0
train_losses = []
val_losses = []
lrs = []

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    global_step = checkpoint["global_step"]
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]
    lrs = checkpoint["lrs"]
    print(f"Loaded checkpoint: step {global_step}", flush=True)

training_set = dataset["train"].select(range(start_idx, end_idx))
validation_set = dataset["validation"]

def tokenize(batch):
  en_texts = [item["en"] for item in batch["translation"]]
  fr_texts = [item["fr"] for item in batch["translation"]]
  return tokenizer.prepare_seq2seq_batch(
      src_texts=en_texts,
      tgt_texts=fr_texts,
      max_length=context_window,
      padding="max_length",
      truncation=True,
      return_tensors="pt"
    )


print("Tokenizing training and validation sets...", flush=True)
tokenized_train = training_set.map(tokenize, batched=True)
tokenized_valid = validation_set.map(tokenize, batched=True)

tokenized_train.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
tokenized_valid.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])

train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(tokenized_valid, batch_size=batch_size)

while(global_step < len(train_loader)):
  model.train()
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

    if global_step % 100 == 0:
      lr = scheduler.get_last_lr()[0]
      avg_loss = running_loss / 100
      print(f"Step {global_step} | Loss: {avg_loss:.4f} | LR: {lr:.6e}", flush=True)
      train_losses.append(avg_loss)
      lrs.append(lr)
      running_loss = 0.0
      # Validation
      print("Running validation...", flush=True)
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
      print(f"Validation Loss: {val_loss:.4f}", flush=True)
      val_losses.append(val_loss)


    if global_step % 500 == 0:

      # Save
      print("Saving checkpoint...", flush=True)
      checkpoint = {
          "model_state_dict": model.state_dict(),
          "optimizer_state_dict": optimizer.state_dict(),
          "scheduler_state_dict": scheduler.state_dict(),
          "global_step": global_step,
          "train_losses": train_losses,
          "val_losses": val_losses,
          "lrs": lrs,
      }
      torch.save(checkpoint, checkpoint_path)
      model.train()



    if global_step >= total_steps:
      break

print("Training complete.", flush=True)


# Plot Results

plt.figure(figsize=(10, 4))
logged_steps = [100 * i for i in range(1, len(train_losses) + 1)]
# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(logged_steps, train_losses, label="Train Loss")
plt.plot(logged_steps, val_losses, label="Validation Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.grid(True)
plt.legend()

# LR Plot
plt.subplot(1, 2, 2)
plt.plot(logged_steps, lrs, label="Learning Rate", color="orange")
plt.xlabel("Step")
plt.ylabel("LR")
plt.title("Learning Rate Schedule")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, "training_progress.png"))
plt.show()
