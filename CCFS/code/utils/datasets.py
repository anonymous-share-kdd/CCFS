import torch
from torch.utils.data import Dataset
import numpy as np



class CCFSDataset(Dataset):
    def __init__(self, df, text_emb, task_type):

        self.X = df.iloc[:, :-1].values.astype(np.float32)
        y_raw = df.iloc[:, -1].values

        if task_type == 'cls':
            self.y = y_raw.astype(np.int64)
        else:
            self.y = y_raw.astype(np.float32)

        self.text_emb = text_emb.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'x_tab': torch.tensor(self.X[idx]),
            'x_text': torch.tensor(self.text_emb[idx]),
            'y': torch.tensor(self.y[idx])
        }