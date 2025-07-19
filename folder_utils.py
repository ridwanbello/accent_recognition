import os
import shutil
from typing import Tuple, List
from pathlib import Path

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

# Function to get the number of files in a folder
def check_folder_len(folder_path: str) -> str:
  """ Checks the number of files in a directory

  Args:
    folder_path: Folder path to be checked

  Returns:
    Message showing the number of files found

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



# Function to move folders to parent
def move_folders_to_parent(folders_to_move, new_parent_folder):
    """ Moves a list of folders to a new parent folder.

    Args:
      folders_to_move (List[str]): List of folder paths to move.
      new_parent_folder (str): Path to the new parent folder.
    """
    os.makedirs(new_parent_folder, exist_ok=True)

    for folder in folders_to_move:
        folder_name = os.path.basename(folder)
        destination = os.path.join(new_parent_folder, folder_name)

        # Move the entire folder
        shutil.move(folder, destination)



# Function to delete last files
def delete_last_files(folder_path, num_files_to_delete=16):
  """ Deletes the last N files in a folder based on modification time.

  Args:
    folder_path (str): Path to the folder.
    num_files_to_delete (int): Number of files to delete from the end.
  """
  try:
      # Get a list of all files in the folder
      files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

      # Sort files by modification time (most recent first)
      files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

      # Select the last num_files_to_delete files
      files_to_delete = files[:num_files_to_delete]

      # Delete the selected files
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

# Function to delete files in a folder
def delete_folder(folder_path):
  """ Deletes a folder and its contents.

  Args:
    folder_path (str): Path to the folder to be deleted.
  """
  if os.path.exists(folder_path):
      try:
          shutil.rmtree(folder_path)
          print(f"Folder '{folder_path}' and its contents deleted successfully.")
      except OSError as e:
          print(f"Error deleting folder {folder_path}: {e}")
  else:
      print(f"Folder '{folder_path}' does not exist.")

# Function to zip a folder
def zipfile(folder_to_zip, output_zip_file):
  """ Zips a folder.

  Args:
    folder_to_zip (str): Path to the folder to zip.
    output_zip_file (str): Path for the output zip file.
  """
  try:
      # Create a zip archive of the folder
      shutil.make_archive(output_zip_file.replace(".zip", ""), 'zip', folder_to_zip)
      print(f"Folder '{folder_to_zip}' successfully zipped to '{output_zip_file}'")

  except FileNotFoundError:
      print(f"Error: The folder '{folder_to_zip}' was not found.")
  except Exception as e:
      print(f"An error occurred: {e}")

# Function to download a zip file to a local machine from Google Colab
def download_zip(zip_file_path):
  """ Downloads a file to the local machine.

  Args:
    zip_file_path (str): Path to the file to download.
  """
  from google.colab import files
  try:
      # Download the zip file
      files.download(zip_file_path)
      print(f"'{zip_file_path}' is ready for download.")

  except FileNotFoundError:
      print(f"Error: The file '{zip_file_path}' was not found.")
  except Exception as e:
      print(f"An error occurred: {e}")
