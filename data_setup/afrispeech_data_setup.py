import soundfile as sf
import itertools
from datasets import load_dataset
from typing import ArrayList, List, Tuple
import os
import shutil
from pathlib import Path

# Install datasets version 2.12.0
def install_hf_datasets_version() -> str:
  """Installs the HF datasets version 2.12.0, which is required to download the AfriSpeech dataset.

  The latest version (4.0.0) as of June 17, 2025 has issues related to dataset format used to upload AfriSpeech.
  
  Args:
    None

  Returns:
    String: A message indicating the version of datasets library installed.
  """
  try:
      import datasets
      if (datasets.__version__ != "2.12.0"):
        !pip install datasets==2.12.0 fsspec==2023.9.2
  except:
      !pip install datasets==2.12.0 fsspec==2023.9.2
      import datasets
  print(f"Datasets version: {datasets.__version__} installed.")


# Download each accent from the afrispeech dataset in batches
def process_afrispeech_batch(accent_list: ArrayList, batch_name: str = "west_africa", split: str = "train", batch_size: int =None) -> str:
    """ Download the Afrispeech dataset in batches using the streaming mode, per accent
        See more information accent by visiting https://huggingface.co/datasets/intronhealth/afrispeech-200

        Args: 
            accent_list: A list of available accents in the Afrispeech dataset.
            batch_name: A name for the folder which belongs to the main class of the accent
            split: The dataset split type. Could be "train", "validation" or "test"
            batch_size: The number of samples to be downloaded

        Returns:
            A print of:
              Statement showing successful processing
              Message showing the path the dataset was saved to
        
    """ 
    base_path = "/content/"
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

# delete files and folder of a particular path
def delete_files(folder_path: Path) -> str: 
  """ Deletes all the files and folders in the folder path recursively.

  Args: 
    folder_path (str or pathlib.Path): Folder path to be deleted

  Returns:
    Successful message showing the folder path deleted

  Raise:
    OSError: If there is an error while deleting the folder
    Error: If the folder does not exist
  """
  if os.path.exists(folder_path):
      try:
          shutil.rmtree(folder_path)
          print(f"Folder '{folder_path}' and its contents deleted successfully.")
      except OSError as e:
          print(f"Error deleting folder {folder_path}: {e}")
  else:
      print(f"Folder '{folder_path}' does not exist.")


# Check the number of samples in a directory
def check_dir_len(folder_path: Path) -> str:
  """ Checks the number of files in a directory

  Args:
    folder_path (str or pathlib.Path): Folder path to be checked

  Returns:
    Message showing the folder path and the number of files found in the folder

  Raise:
    FileNotFoundError: If the folder path was not found
    Exception: If there is any error occured while assessing or computing the number of files
  """
  try:
      entries = os.listdir(folder_path)
      files = [entry for entry in entries if os.path.isfile(os.path.join(folder_path, entry))]
      num_files = len(files)
      print(f"The number of files in '{folder_path}' is: {num_files}")
  except FileNotFoundError:
      print(f"Error: The folder '{folder_path}' was not found.")
  except Exception as e:
      print(f"An error occurred: {e}")

# Delete some audio files
def delete_last_files(folder_path: Path, num_files_to_delete: int = 16) -> str:
  """ Deletes the last number of files, based on the number of files entered in the second argument
  
  Args:
    folder_path (str or pathlib.Path): Folder path where the files will be deleted
    num_files_to_delete: The number of files to be deleted

  Returns:
    A print of:
        Message showing the number of files deleted from the folder path
        List of files deleted

  Raise:
    OSError: If any error occurs while deleting
    FileNotFoundError: If the folder path does not exist
    Exception: If there is any error while deleting the files
  """
  try:
      files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
      files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
      files_to_delete = files[:num_files_to_delete]
      print(f"Deleting the last {num_files_to_delete} files from '{folder_path}':")
      for file_path in files_to_delete:
          try:
              os.remove(file_path)
              print(f"Deleted: {file_path}")
          except OSError as e:
              print(f"Error deleting {file_path}: {e}")
  except FileNotFoundError:
      print(f"Error: The folder '{folder_path}' was not found.")
  except Exception as e:
      print(f"An error occurred: {e}")

# Move each accent class folder to a parent folder for the classification format
def move_folders_to_parent(folders_to_move: ArrayList, new_parent_folder: str):
    """ Create the new parent folder if not exist and moves all specified folders to the newly created folder.
    
    Args:
      folders_to_move: An array consisting of all the accent folder paths to be moved to the parent folder
      new_parent_folder: Name of the new parent folder to be created

    Returns:
      A message showing the successful moving of the folders
    
    Raise:
      Exception: If there is any error while moving the folders
    """
    os.makedirs(new_parent_folder, exist_ok=True)

    for folder in folders_to_move:
        try:
            folder_name = os.path.basename(folder)
            destination = os.path.join(new_parent_folder, folder_name)
            shutil.move(folder, destination)
        except Exception as e:
           print(f"Error moving the folders {e}")
    print("Folders moved successfully")

# View information on number of files in each folders and subsequent subfolders
def walk_through_dir(folder_path: Path):
  """ Walks through the folder and returns its contents.

  Args:
    folder_path (str or pathlib.Path): Folder we want to view

  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(folder_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} samples in '{dirpath}'.")

def prepare_audio_list(folder_path: Path) -> Tuple[List, List]:
  """ Get the audio and list
  Args:
    folder_path (str or pathlib.Path): Folder we want to get the aduio files from
  
  Returns:
    A tuple of list of audios and labels
  """
  audio_files = list(folder_path.glob("*/*.wav"))
  data = [(file, file.parent.stem) for file in audio_files]
  audio_list, label_list = zip(*data)
  return audio_list, label_list

   
   