import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class DataLoader(DataLoader):
    def __init__(self, data_dir, batch_size, device):
        data = np.load(data_dir).astype('float32')
        data = torch.from_numpy(data).to(device)
        data_set = TensorDataset(data)

        super().__init__(dataset=data_set,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)
