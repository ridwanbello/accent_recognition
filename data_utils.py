from typing import List, Tuple
from pathlib import Path

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

