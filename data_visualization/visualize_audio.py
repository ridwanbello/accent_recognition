import random
import librosa
import librosa.display
import IPython.display as ipd
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import torch
import matplotlib.pyplot as plt


# Set seed
# random.seed(43)
def show_sample_audio(audio_path_list):
  """ Show a sample audio with shape, size and sample rate
  
  Args:
  audio_path_list: List of audio paths
  
  """
  random_audio_path = random.choice(audio_path_list)
  audio_label = random_audio_path.parent.stem
  signal, sr = torchaudio.load(random_audio_path)
  number_of_samples = signal.shape[1]
  audio_duration = number_of_samples / sr
  # Print metadata
  print(f"Random audio path: {random_audio_path}")
  print(f"Audio Label: {audio_label}")
  print(f"Signal Shape: {signal.shape}")
  print(f"Sample Rate: {sr}")
  print(f"Audio Duration: {round(audio_duration)} seconds")

  # Display the audio
  ipd.Audio(data=signal, rate=sr)



# Plot the waveform of audio
def plot_waveform(waveform, sr, title="Waveform", ax=None):
    """ The waveform of an audio signal 
    Args: 
    waveform: the signal after loading using librosa or torchaudio
    sr: sample rate
    title: Title of the wave plot
    ax: axis of the plot
    """
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


# Plot either spectrogram or mfcc
def plot_transformation(specgram, title=None, ylabel="freq_bin", ax=None):
    """ The plot showing spectrogram 
    Args: Spectrogram signal
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

# Plot showing n samples of audio with spectrogram
def plot_multiple_audio_with_spectrogram(audio_path_list, n = 10):
  """The plot for visualization of audio waveform alongside spectrograms"""
  random_audio_paths = random.sample(audio_path_list, k = n)
  for audio_path in random_audio_paths:
    audio_label = audio_path.parent.stem
    signal, sr = torchaudio.load(audio_path)
    spectrogram = T.Spectrogram(n_fft=512)
    spec = spectrogram(signal)
    fig, axs = plt.subplots(1, 2, figsize=(15, 3))
    plot_waveform(
        signal, sr,
        title="Original waveform",
        ax=axs[0])
    plot_transformation(
        spec[0],
        title="Spectrogram",
        ax=axs[1])
    fig.tight_layout()
    plt.suptitle(f"Label: {audio_label.capitalize()}")

# Define MFCC transformation
def mfcc_transform(signal, sr):
    transform = T.MFCC(
            sample_rate=sr,
            n_mfcc=256,
            melkwargs={
                "n_fft": 2048,
                "n_mels": 256,
                "hop_length": 512,
                "mel_scale": "htk",
            },
        )
    return transform(signal)

def spectrogram_transform(signal, sr):
  mel_spectrogram = torchaudio.transforms.MelSpectrogram(
      sample_rate=sr,
      n_fft=1024,
      hop_length=512,
      n_mels=64
  )
  return mel_spectrogram(signal)


# Plot showing samples of audio, spectrogram, and MFCC
def plot_multiple_audio_with_spectrogram_and_mfcc(audio_path_list, n = 10):
  """ The plot for visualization of audio waveform alongside spectrograms and MFCCs """
  random_audio_paths = random.sample(audio_path_list, k = n)
  for audio_path in random_audio_paths:
    audio_label = audio_path.parent.stem
    signal, sr = torchaudio.load(audio_path)
    spec = spectrogram_transform(signal, sr)
    mfcc = mfcc_transform(signal, sr)
    # Plot the graph
    fig, axs = plt.subplots(1, 3, figsize=(20, 3))
    plot_waveform(
        signal, sr,
        title=f"Original Waveform ({audio_label.capitalize()})",
        ax=axs[0])
    plot_transformation(
        spec[0],
        title=f"Spectrogram ({audio_label.capitalize()})",
        ax=axs[1])
    plot_transformation(
      mfcc[0],
      title=f"MFCC ({audio_label.capitalize()})",
      ax=axs[2])