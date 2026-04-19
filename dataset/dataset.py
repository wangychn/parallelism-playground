from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer

import torch
from torch.utils.data import DataLoader, Dataset

class MoEDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size
    
    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + 1 + self.block_size]
        return x, y

def to_flat_array(data, tokenizer):
    tokenized = []
    for text in data:
        ids = tokenizer(
            text["text"],
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )["input_ids"]
        ids.append(tokenizer.eos_token_id)
        tokenized.extend(ids)
    
    return torch.tensor(tokenized, dtype=torch.long)
        
def load_hf_dataset(dataset, size=10_000):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    train_size = int(size * 0.7)

    match dataset:
        case "openwebtext":
            # https://huggingface.co/datasets/Skylion007/openwebtext

            print(get_dataset_split_names(dataset))

            # https://huggingface.co/docs/datasets/en/stream
            data_iter = load_dataset(dataset, split="train", streaming=True)
            
            data = data_iter.take(size)
            
            # splitting logic
            train = data.take(train_size)
            val = data.skip(train_size).take(size - train_size)
            return to_flat_array(train, tokenizer), to_flat_array(val, tokenizer)
            
        case _:
            print("Pick a supported dataset")


    # return train_data, val_data


def build_train_val_loaders(train, val, batch_size, block_size):
    """
    https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """

    train_loader = DataLoader(
        MoEDataset(train, block_size),
        batch_size=batch_size
    )

    val_loader = DataLoader(
        MoEDataset(val, block_size),
        batch_size=batch_size
    )

    return train_loader, val_loader
    


if __name__ == "__main__":
    t, v = load_hf_dataset("openwebtext")
    tl, vl = build_train_val_loaders(t, v, 8, 8)

    x, y = next(iter(tl))

    print(x)
    print(y)

