import os
import torch
import pandas as pd
from transformers import BertTokenizerFast
from cosmosqa.data_loader.dataloader import create_data_loader
from cosmosqa.model.bert_ma import BertMultiwayMatch

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PRE_TRAINED_MODEL_NAME = "bert-large-uncased"

MAX_LEN = 220
BATCH_SIZE = 2
EPOCHS = 1

tokenizer = BertTokenizerFast.from_pretrained(PRE_TRAINED_MODEL_NAME)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


config = {"hidden_size": 1024, "hidden_dropout_prob": 0.1}
df_train = pd.read_csv("./cosmosqa/data/train_sample.csv")
df_valid = pd.read_csv("./cosmosqa/data/valid_sample.csv")

train_data_loader = create_data_loader(
    df=df_train, tokenizer=tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE
)
valid_data_loader = create_data_loader(
    df=df_valid, tokenizer=tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE
)

model = BertMultiwayMatch.from_pretrained(PRE_TRAINED_MODEL_NAME, num_choices=4)

for d in train_data_loader:
    input_ids = d["input_ids"]
    attention_mask = d["attention_mask"]
    token_type_ids = d["token_type_ids"]
    c_len = d["c_len"]
    q_len = d["q_len"]
    a_len = d["a_len"]
    targets = d["target"].type(dtype=torch.long)

    loss, logits = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        doc_len=c_len,
        ques_len=q_len,
        option_len=a_len,
        labels=targets,
    )
    print("loss")
    print(loss)
    print("logits")
    print(logits)
