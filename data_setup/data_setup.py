import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os
from torch.utils.data import random_split, DataLoader



# Convert to Pytorch Dataset format
class AccentDataset(Dataset):
  """Convert dataset into classification format"""
  def __init__(self,
                 audio_path,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.paths = list(audio_path.glob("*/*.wav"))
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = self._find_classes(audio_path)

  def __len__(self):
        return len(self.paths)

  def __getitem__(self, index):
      audio_sample_path = self.paths[index]
      signal, sr = torchaudio.load(audio_sample_path)
      signal = signal.to(self.device)
      signal = self._resample_if_necessary(signal, sr)
      signal = self._mix_down_if_necessary(signal)
      signal = self._cut_if_necessary(signal)
      signal = self._right_pad_if_necessary(signal)
      signal = self.transformation(signal).to(self.device)
      class_name  = self.paths[index].parent.name
      class_idx = self.class_to_idx[class_name]
      return signal, class_idx

  def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

  def _right_pad_if_necessary(self, signal):
      length_signal = signal.shape[1]
      if length_signal < self.num_samples:
          num_missing_samples = self.num_samples - length_signal
          last_dim_padding = (0, num_missing_samples)
          signal = torch.nn.functional.pad(signal, last_dim_padding)
      return signal

  def _resample_if_necessary(self, signal, sr):
      if sr != self.target_sample_rate:
          resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
          signal = resampler(signal)
      return signal

  def _mix_down_if_necessary(self, signal):
      if signal.shape[0] > 1:
          signal = torch.mean(signal, dim=0, keepdim=True)
      return signal
  
  def _find_classes(audio_path):
    classes = sorted(entry.name for entry in os.scandir(audio_path) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {audio_path}.")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
  


# Splitting the data return in Dataset format into 80 for training, 10 for validation and 10 for Testing
def train_test_split_80_10_10(speech_data):
  """ Split data into 80-10-10
  Args: speech_data: data should be a series of (audio,label) format
  """
  train_size = int(0.8 * len(speech_data))
  val_size = len(speech_data) - train_size
  test_size = val_size = int(val_size / 2)

  generator = torch.Generator().manual_seed(42)
  train_subset, val_subset, test_subset = random_split(speech_data, [train_size, val_size, test_size], generator=generator)

  train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
  test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

  return train_loader, val_loader, test_loader

# Sample usage
# train_loader, val_loader, test_loader = train_test_split_80_10_10(speech_data)
# len(train_loader), len(val_loader), len(test_loader)
# signal, label = next(iter(train_loader))