from torch.utils.data import DataLoader
from cosmosqa.data_loader.dataset import CosmosQADataset


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = CosmosQADataset(
        df=df,
        targets=df["label"].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=1)
