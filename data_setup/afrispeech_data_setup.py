import soundfile as sf
import itertools
from datasets import load_dataset
import os
from typing import List, Tuple
from pathlib import Path

# Function to process Afrispeech batch
def process_afrispeech_batch(accent_list: List[str], batch_name: str = "west_africa", split: str = "train", batch_size: int =None):
    """ Download the Afrispeech dataset in batches using the streaming mode, per accent
        Requires HF datasets==2.12.0 and fsspec==2023.9.2
        See more information accent by visiting https://huggingface.co/datasets/intronhealth/afrispeech-200

        Args:
            accent_list: A list of available accents in the Afrispeech dataset.
            batch_name: A name for the folder which belongs to the main class of the accent
            split: The dataset split type. Could be "train", "validation" or "test"
            batch_size: The number of samples to be downloaded

        Returns:
            Data saved in the folder
            Print statement showing successful processing
            Message showing the path the dataset was saved to

    """
    base_path = "/content/afrispeech"
    main_path = os.path.join(base_path, batch_name)
    os.makedirs(main_path, exist_ok=True)

    for accent in accent_list:
      print(f"Processing accent: {accent}")
      # Load dataset in streaming mode with caching disabled
      ds = load_dataset("tobiolatunji/afrispeech-200", accent, split=split, streaming=True, cache_dir=None)

      for i, example in enumerate(itertools.islice(ds, 0, batch_size)):
        audio_array = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]
        accent = example["accent"]
        i = i + 1
        sf.write(f"{main_path}/{accent}_{i}.wav", audio_array, sampling_rate) # Use main_path for writing files

      print(f"Finished processing {accent}")

    print(f"Dataset saved to: {main_path}")

# Function to walk through directory
def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.
  Args:
    dir_path (str or pathlib.Path): target directory

  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} audio samples in '{dirpath}'.")

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


