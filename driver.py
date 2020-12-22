import os
import torch
import time
import pandas as pd
from transformers import (
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
    AdamW,
    logging,
)
from torch.nn import CrossEntropyLoss
from collections import defaultdict
from cosmosqa.data_loader.dataloader import create_data_loader
from cosmosqa.trainer.train import train_epoch, eval_model
from cosmosqa.model.bert_ma import BertMultiwayMatch

logging.set_verbosity_error()

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PRE_TRAINED_MODEL_NAME = "bert-large-uncased"

N_TRAIN_SAMPLES = 10000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if str(device) == "cpu":
    MAX_LEN = 128
    BATCH_SIZE = 2
    EPOCHS = 1
    lr = 2e-5
    adam_epsilon = 1e-8
else:
    MAX_LEN = 128
    BATCH_SIZE = 8
    EPOCHS = 10
    lr = 2e-5
    adam_epsilon = 1e-8

print("Parameters: ")
print(f"MAX_LEN: {MAX_LEN}")
print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"EPOCHS: {EPOCHS}")
print(f"lr: {lr}")
print(f"adam_epsilon: {adam_epsilon}")

tokenizer = BertTokenizerFast.from_pretrained(PRE_TRAINED_MODEL_NAME)

if str(device) == "cpu":
    df_train = pd.read_csv("./cosmosqa/data/train_sample.csv")
    df_valid = pd.read_csv("./cosmosqa/data/valid_sample.csv")
else:
    df_train = pd.read_csv("./cosmosqa/data/train.csv").head(N_TRAIN_SAMPLES)
    df_valid = pd.read_csv("./cosmosqa/data/valid.csv")

train_data_loader = create_data_loader(
    df=df_train, tokenizer=tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE
)
valid_data_loader = create_data_loader(
    df=df_valid, tokenizer=tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE
)

num_warmup_steps = 0
num_training_steps = len(train_data_loader) * EPOCHS


model = BertMultiwayMatch.from_pretrained(PRE_TRAINED_MODEL_NAME, num_choices=4)
model = model.to(device)
optimizer = AdamW(
    model.parameters(), lr=lr, eps=adam_epsilon, correct_bias=False
)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)  # PyTorch scheduler
loss_fn = CrossEntropyLoss().to(device)
history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
    start_time = time.time()
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 10)
    train_acc, train_loss = train_epoch(
        model, train_data_loader, loss_fn, optimizer, device, scheduler
    )
    print(f"Train loss {train_loss} accuracy {train_acc}")

    val_acc, val_loss = eval_model(model, valid_data_loader, loss_fn, device)
    print(f"Val  loss {val_loss} accuracy {val_acc}")
    print()

    history["train_acc"].append(train_acc)
    history["train_loss"].append(train_loss)
    history["val_acc"].append(val_acc)
    history["val_loss"].append(val_loss)
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), "best_model_state.bin")
        best_accuracy = val_acc
    end_time = time.time()
    total_epoch_time = end_time - start_time
    total_epoch_time = round(total_epoch_time, 2)
    print(f"Epoch {epoch + 1}/{EPOCHS} took {total_epoch_time} secs")


print("History:")
print(history)
