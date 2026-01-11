import torch
from torch.utils.data import Dataset
import numpy as np

class TrajectoryDataset(Dataset):
    def __init__(self, noisy_file, clean_file):
        # Load noisy and clean trajectories
        self.noisy = np.load(noisy_file)
        self.clean = np.load(clean_file)
        
        # Ensure shapes match
        if len(self.noisy) != len(self.clean):
            raise ValueError("Noisy and clean data must have the same length")

    def __len__(self):
        return int(len(self.noisy))  # <- MUST return a non-negative integer

    def __getitem__(self, idx):
        x = self.noisy[idx]
        y = self.clean[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
