import os
import shutil
from pathlib import Path
from typing import List, Tuple
import tarfile

def download_accentdb(accent_list: List) -> Tuple[List, List]:
    """ Download the AccentDB dataset from Google Drive and return as the List in tuples
        Available accent list includes "bangla", "malayalam", "odiya", "telugu", "indian", "australian", "british", "american", and "welsh"

        Args:
            accent_list: List of accents to be included in the form of an array. Samples like ["indian", "american"].

        Returns:
            A tuple of audio and label
            A print of:
                Information about the number of files in folders and subfolders
    """
    def check_if_its_the_right_input_format(data):
        return isinstance(data, list) and all(isinstance(item, str) for item in data)

    if(check_if_its_the_right_input_format(accent_list) == False):
        return f"Data not in the right format. Shoul be like ['american', 'indian']"

    try:
        import gdown
    except:
        !pip install gdown
        import gdown

    data_path = Path("data/")

    if data_path.is_dir():
        print(f"{data_path} directory exists.")
    else:
        print(f"Did not find {data_path} directory, creating one...")
        data_path.mkdir(parents=True, exist_ok=True)
    # Google Drive link to download the data
    url = 'https://drive.google.com/uc?id=1i2UOz9-M8cIPzkIapUQPFxB-sRE29xvU'
    output_filename = data_path / "accentDB_extended.tar.gz"
    gdown.download(url, str(output_filename), quiet=False)
    zipped_path = Path(data_path / "accentDB_extended.tar.gz")
    extracted_data = Path(data_path / "extracted_data")
    # Unzip the data
    with tarfile.open(zipped_path, 'r:gz') as tar:
        tar.extractall(path=extracted_data)

    audio_data = Path(extracted_data / "data")

    for dirpath, dirnames, filenames in os.walk(audio_data):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    accent_data = data_path / "accent_data"
    # The new folder that will house the main data will be called "accent_data"
    os.makedirs(accent_data, exist_ok=True)
    # Move folders in selected accented to a new folder called "accent_data"
    for folder in accent_list:
        src_path = os.path.join(audio_data, folder)
        dest_path = os.path.join(accent_data, folder)

        if os.path.exists(src_path):
            shutil.copytree(src_path, dest_path)
            print(f"Moved : {folder}")
        else:
            print(f"Folder not found: {folder}")

    # Delete extra speaker information in american folders to make each accent atleast 2
    def delete_american_folders(american_folders_to_delete):
        for i in range(3,9):
            folder_path = os.path.join(american_folders_to_delete, f"speaker_0{i}")
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                print(f"Deleted: {folder_path}")
            else:
                print("Folder does not exist.")

    american_folders_to_delete = Path(accent_data / "american")
    delete_american_folders(american_folders_to_delete)

    # Prepare and return audio list
    audio_files = list(accent_data.glob("*/*/*.wav"))
    data = [(file, file.parent.parent.stem) for file in audio_files]
    audio_list, label_list = zip(*data)
    print(f"There are {len(audio_list)} audio samples returned.")
    return audio_list, label_list

# Commit 3:  Added extra comment, minor error changes from data_path to accent_data