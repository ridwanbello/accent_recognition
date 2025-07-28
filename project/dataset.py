import os
import librosa
import numpy as np
from torch.utils.data import Dataset
from config import Config
import torch 

class AccentDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        waveform, _ = librosa.load(path, sr=Config.SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=waveform, sr=Config.SAMPLE_RATE, n_mfcc=Config.N_MFCC)

        # Padding or truncating
        if mfcc.shape[1] < Config.MAX_LEN:
            pad_width = Config.MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :Config.MAX_LEN]

        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(label)
