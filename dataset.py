import numpy as np
import torch
from torch.utils.data import Dataset

def create_sequences(features: np.ndarray, targets: np.ndarray, seq_len: int):
    X, Y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i : i + seq_len])
        Y.append(targets[i + seq_len])
    return np.array(X), np.array(Y)

class StockDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
