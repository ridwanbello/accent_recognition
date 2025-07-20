import os
from typing import Tuple, Dict, List
from pathlib import Path
import torch
import pandas as pd
import torchaudio
from torch.utils.data import Dataset, random_split, DataLoader

def prepare_audio_list(folder_path: Path) -> Tuple[List, List]:
  """ Get the audio and list
  Args:
    folder_path (str or pathlib.Path): Folder we want to get the aduio files from
  
  Returns:
    A tuple of list of audios and labels
  """
  folder_path = Path(folder_path)
  audio_files = list(folder_path.glob("*/*.wav"))
  data = [(file, file.parent.stem) for file in audio_files]
  audio_list, label_list = zip(*data)
  return audio_list, label_list

# Return the target labels in a dictionary format for efficient mapping
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.

    Assumes the folder is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))

    Raise: 
        FileNotFoundError: In case the file does not exist

    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


# Convert to Pytorch Dataset Classifation format
class AccentDataset(Dataset):
  """Convert dataset into classification format"""
  def __init__(self,
                 data_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.paths = list(data_dir.glob("*/*.wav"))
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(data_dir)

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

# Convert audio signals to Spectrogram
def spectrogram_transformation(sample_rate: int = 44100):
  """ Convert audio signals into Spectrogram. """
  mel_spectrogram = torchaudio.transforms.MelSpectrogram(
      sample_rate=sample_rate,
      n_fft=1024,
      hop_length=512,
      n_mels=64
  )
  return mel_spectrogram

# Convert audio signals to MFCC
def mfcc_transformation(sample_rate: int = 44100):
  """ Convert audio signals into MFCC. """
  mfcc = torchaudio.transforms.MFCC(
  sample_rate=sample_rate,
  n_mfcc=13,
  melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
  )
  return mfcc

# Split into 80-10-10
def train_test_split_80_10_10(speech_data):
  """ Splits data into 80% for Training, 10% for Validation, and 10% for Testing """
  train_size = int(0.8 * len(speech_data))
  val_size = len(speech_data) - train_size
  test_size = val_size = int(val_size / 2)

  # Setting random seed to 42 for reproducibility
  generator = torch.Generator().manual_seed(42)
  train_subset, val_subset, test_subset = random_split(speech_data, [train_size, val_size, test_size], generator=generator)

  train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
  test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

  return train_loader, val_loader, test_loader

