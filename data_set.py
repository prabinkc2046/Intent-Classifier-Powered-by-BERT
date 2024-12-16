from torch.utils.data import Dataset
import torch

from tokenize_data import tokenize_data

class IntentDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer):
        self.tokens = tokenize_data(tokenizer, sentences)
        self.labels = torch.tensor(labels.values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokens["input_ids"][idx],
            "attention_mask": self.tokens["attention_mask"][idx],
            "labels": self.labels[idx]
        }

