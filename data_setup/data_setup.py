import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os
from torch.utils.data import random_split, DataLoader
import random
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain
import numpy as np

# Splitting the data return in Dataset format into 80 for training, 10 for validation and 10 for Testing
def train_test_split_80_10_10(speech_data):
    """
    Splits a dataset into 80% training, 10% validation, and 10% testing sets.
    """
    train_size = int(0.8 * len(speech_data))
    val_size = int(0.1 * len(speech_data))
    # To ensure the splits add up to the total length
    test_size = len(speech_data) - train_size - val_size  

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = torch.utils.data.random_split(speech_data, [train_size, val_size, test_size], generator=generator)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader

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
  
  def _find_classes(self, audio_path):
    classes = sorted(entry.name for entry in os.scandir(audio_path) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {audio_path}.")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
  

# Accent Dataset With Mixin
import random

class AccentDatasetWithMixinAfter(Dataset):
    """Convert dataset into classification format with optional same-class mixin"""
    def __init__(self,
                 audio_path,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device,
                 mix_prob=0.5): 
        self.paths = list(audio_path.glob("*/*.wav"))
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.mix_prob = mix_prob
        self.classes, self.class_to_idx = self._find_classes(audio_path)

        self.class_idx_to_indices = self._build_class_indices()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        audio_sample_path = self.paths[index]
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._preprocess(signal, sr)
        signal = self.transformation(signal).to(self.device)

        class_name = audio_sample_path.parent.name
        class_idx = self.class_to_idx[class_name]

        # Do same-class mixin with probability
        if random.random() < self.mix_prob:
            # Pick another sample from same class (but different index)
            same_class_indices = self.class_idx_to_indices[class_idx]
            alt_index = index
            while alt_index == index:
                alt_index = random.choice(same_class_indices)
            alt_sample_path = self.paths[alt_index]

            alt_signal, alt_sr = torchaudio.load(alt_sample_path)
            alt_signal = alt_signal.to(self.device)
            alt_signal = self._preprocess(alt_signal, alt_sr)
            alt_signal = self.transformation(alt_signal).to(self.device)

            # Ensure both tensors have same shape
            min_len = min(signal.shape[-1], alt_signal.shape[-1])
            signal = signal[..., :min_len]
            alt_signal = alt_signal[..., :min_len]

            # Perform mixin (simple average)
            lam = torch.distributions.Beta(0.4, 0.4).sample().item()
            signal = lam * signal + (1 - lam) * alt_signal

        return signal, class_idx

    def _preprocess(self, signal, sr):
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        return signal

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

    def _find_classes(self, audio_path):
        classes = sorted(entry.name for entry in os.scandir(audio_path) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in {audio_path}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _build_class_indices(self):
        class_indices = {i: [] for i in range(len(self.classes))}
        for idx, path in enumerate(self.paths):
            class_name = path.parent.name
            class_idx = self.class_to_idx[class_name]
            class_indices[class_idx].append(idx)
        return class_indices


# AccentDatasetWithMixinAfter

class AccentDatasetWithMixin(Dataset):
    """Same-class mixin done BEFORE transformation."""
    def __init__(self,
                 audio_path,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device,
                 mix_prob=0.5): 
        self.paths = list(audio_path.glob("*/*.wav"))
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.mix_prob = mix_prob
        self.classes, self.class_to_idx = self._find_classes(audio_path)

        self.class_idx_to_indices = self._build_class_indices()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Load and preprocess main sample
        audio_sample_path = self.paths[index]
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._preprocess(signal, sr)

        class_name = audio_sample_path.parent.name
        class_idx = self.class_to_idx[class_name]

        # Same-class mixin BEFORE transformation
        if random.random() < self.mix_prob:
            same_class_indices = self.class_idx_to_indices[class_idx]
            alt_index = index
            while alt_index == index:
                alt_index = random.choice(same_class_indices)
            alt_sample_path = self.paths[alt_index]

            alt_signal, alt_sr = torchaudio.load(alt_sample_path)
            alt_signal = alt_signal.to(self.device)
            alt_signal = self._preprocess(alt_signal, alt_sr)

            # Match lengths
            min_len = min(signal.shape[-1], alt_signal.shape[-1])
            signal = signal[..., :min_len]
            alt_signal = alt_signal[..., :min_len]

            # Blend raw waveforms
            lam = torch.distributions.Beta(0.4, 0.4).sample().item()
            signal = lam * signal + (1 - lam) * alt_signal

        # Now transform once
        signal = self.transformation(signal).to(self.device)

        return signal, class_idx

    def _preprocess(self, signal, sr):
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        return signal

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

    def _find_classes(self, audio_path):
        classes = sorted(entry.name for entry in os.scandir(audio_path) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in {audio_path}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _build_class_indices(self):
        class_indices = {i: [] for i in range(len(self.classes))}
        for idx, path in enumerate(self.paths):
            class_name = path.parent.name
            class_idx = self.class_to_idx[class_name]
            class_indices[class_idx].append(idx)
        return class_indices
    
# Accent Dataset with Classic Data Augmentation
class AccentDatasetWithAug(Dataset):
    """Dataset with audiomentations augmentation for accent recognition."""

    def __init__(self,
                 audio_path,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device,
                 apply_augmentation=True):
        self.paths = list(audio_path.glob("*/*.wav"))
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.apply_augmentation = apply_augmentation
        self.classes, self.class_to_idx = self._find_classes(audio_path)

        # Augmentation pipeline: only applied with 50% probability
        self.augment = Compose([
            Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
                Gain(min_gain_db=-6, max_gain_db=6, p=0.3)
            ], p=0.5)  # 50% chance to apply any augmentation at all
        ])

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

        # Apply augmentation if enabled
        if self.apply_augmentation:
            signal_np = signal.squeeze(0).cpu().numpy().astype(np.float32)  # mono → numpy
            signal_np = self.augment(samples=signal_np, sample_rate=self.target_sample_rate)
            signal = torch.tensor(signal_np, dtype=torch.float32).unsqueeze(0).to(self.device)

        signal = self.transformation(signal).to(self.device)

        class_name = audio_sample_path.parent.name
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

    def _find_classes(self, audio_path):
        classes = sorted(entry.name for entry in os.scandir(audio_path) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in {audio_path}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
# With Mixin after Data Augmentation
class AccentDatasetWithMixinAug(Dataset):
    """Dataset with same-class mixin + audiomentations augmentation."""

    def __init__(self,
                 audio_path,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device,
                 mix_prob=0.5,
                 apply_augmentation=True):
        self.paths = list(audio_path.glob("*/*.wav"))
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.mix_prob = mix_prob
        self.apply_augmentation = apply_augmentation
        self.classes, self.class_to_idx = self._find_classes(audio_path)

        self.class_idx_to_indices = self._build_class_indices()

        # Augmentation pipeline: 50% chance overall
        self.augment = Compose([
            Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
                PitchShift(min_semitones=-2, max_semitones=2, p=0.3),  # subtle for speech
                Gain(min_gain_db=-6, max_gain_db=6, p=0.3)
            ], p=0.5)  # global pipeline probability
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Load main sample
        audio_sample_path = self.paths[index]
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._preprocess(signal, sr)

        class_name = audio_sample_path.parent.name
        class_idx = self.class_to_idx[class_name]

        # ---- SAME-CLASS MIXIN ----
        if random.random() < self.mix_prob:
            same_class_indices = self.class_idx_to_indices[class_idx]
            alt_index = index
            while alt_index == index:
                alt_index = random.choice(same_class_indices)
            alt_sample_path = self.paths[alt_index]
            alt_signal, alt_sr = torchaudio.load(alt_sample_path)
            alt_signal = alt_signal.to(self.device)
            alt_signal = self._preprocess(alt_signal, alt_sr)

            # Ensure same length
            min_len = min(signal.shape[-1], alt_signal.shape[-1])
            signal = signal[..., :min_len]
            alt_signal = alt_signal[..., :min_len]

            lam = torch.distributions.Beta(0.4, 0.4).sample().item()
            signal = lam * signal + (1 - lam) * alt_signal

        # ---- AUDIO AUGMENTATION ----
        if self.apply_augmentation:
            signal_np = signal.squeeze(0).cpu().numpy().astype(np.float32)
            signal_np = self.augment(samples=signal_np, sample_rate=self.target_sample_rate)
            signal = torch.tensor(signal_np, dtype=torch.float32).unsqueeze(0).to(self.device)

        # ---- FEATURE TRANSFORMATION ----
        signal = self.transformation(signal).to(self.device)

        return signal, class_idx

    # -----------------------------
    # Preprocessing helpers
    # -----------------------------
    def _preprocess(self, signal, sr):
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        return signal

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

    def _find_classes(self, audio_path):
        classes = sorted(entry.name for entry in os.scandir(audio_path) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in {audio_path}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _build_class_indices(self):
        class_indices = {i: [] for i in range(len(self.classes))}
        for idx, path in enumerate(self.paths):
            class_name = path.parent.name
            class_idx = self.class_to_idx[class_name]
            class_indices[class_idx].append(idx)
        return class_indices


class AccentDatasetWithMixinSpecAug(Dataset):
    """Same-class mixin BEFORE spectrogram, then SpecAugment."""
    def __init__(self,
                 audio_path,
                 transformation,      # e.g., MelSpectrogram
                 spec_augment,        # e.g., torchaudio.transforms.TimeMasking
                 target_sample_rate,
                 num_samples,
                 device,
                 mix_prob=0.5):

        self.paths = list(audio_path.glob("*/*.wav"))
        self.device = device
        self.transformation = transformation.to(self.device)  # waveform → spectrogram
        self.spec_augment = spec_augment.to(self.device)      # spectrogram → augmented spectrogram
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.mix_prob = mix_prob
        self.classes, self.class_to_idx = self._find_classes(audio_path)
        self.class_idx_to_indices = self._build_class_indices()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Load & preprocess waveform
        audio_sample_path = self.paths[index]
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._preprocess(signal, sr)

        class_name = audio_sample_path.parent.name
        class_idx = self.class_to_idx[class_name]

        # Same-class mixin BEFORE spectrogram
        if random.random() < self.mix_prob:
            same_class_indices = self.class_idx_to_indices[class_idx]
            alt_index = index
            while alt_index == index:
                alt_index = random.choice(same_class_indices)
            alt_path = self.paths[alt_index]
            alt_signal, alt_sr = torchaudio.load(alt_path)
            alt_signal = alt_signal.to(self.device)
            alt_signal = self._preprocess(alt_signal, alt_sr)

            # Match lengths
            min_len = min(signal.shape[-1], alt_signal.shape[-1])
            signal = signal[..., :min_len]
            alt_signal = alt_signal[..., :min_len]

            # Blend waveforms
            lam = torch.distributions.Beta(0.4, 0.4).sample().item()
            signal = lam * signal + (1 - lam) * alt_signal

        # Convert to spectrogram
        spec = self.transformation(signal)

        # Apply SpecAugment
        spec = self.spec_augment(spec)

        return spec, class_idx

    def _preprocess(self, signal, sr):
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        return signal

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

    def _find_classes(self, audio_path):
        classes = sorted(entry.name for entry in os.scandir(audio_path) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in {audio_path}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _build_class_indices(self):
        class_indices = {i: [] for i in range(len(self.classes))}
        for idx, path in enumerate(self.paths):
            class_name = path.parent.name
            class_idx = self.class_to_idx[class_name]
            class_indices[class_idx].append(idx)
        return class_indices