import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List

# Download SpeechArchiveAccent database from Kaggle
def download_speechaccentarchive_data(accent_list: List) -> Tuple[List, List]:
    """Downloads the Speech Archive Accent dataset from Kaggle, selects the list of accents and convert into appropriate format for Pytorch DataLoader
    Steps:
        1. Download from Kagglehub
        2. Get the message read by the speakers
        3. Convert the speech name and label into DataFrame format
        4. Filter out the selected labels
        5. Show the data distribution
        6. Prepare into audio list and corresponding label list
    More information on the accents can be found here: https://www.kaggle.com/datasets/rtatman/speech-accent-archive

    Args:
      accent_list: List of accents to be included in the form of an array. Samples like ["arabic", "english"]. 

    Returns:
        A tuple of audio and label
        A print of:
            List of files
            Message read by the speakers
            The first five rows of the data
            Plot of the label distribution
            Length of the final audio and label files

    """
    def check_if_its_the_right_input_format(data):
        return isinstance(data, list) and all(isinstance(item, str) for item in data)
    
    if(check_if_its_the_right_input_format(accent_list) == False):
        return f"Data not in the right format e.g {["arabic", "english"]}"

    try:
        import kagglehub
    except:
        !pip install kagglehub
    print(f"Kaggle version: {kagglehub.__version__} installed.")
    try:
        data_path = kagglehub.dataset_download("rtatman/speech-accent-archive")
    except Exception as e:
        print(f"An error occured while downloading the dataset from Kaggle: {e}")

    print("Files in the downloaded path:", os.listdir(data_path))
    
    # Get the message read by the speakers
    text_path = os.path.join(data_path, "reading-passage.txt")
    with open(text_path, 'r') as f:
        text_content = f.read()
    print(f" Message read by the speakers: {text_content}")

    # The audio files are in the "recordings/recordings" path
    audio_path = os.path.join(data_path, "recordings/recordings")
    audio_list = os.listdir(audio_path)
    # Convert ot DataFrame format for easy filtering
    df = pd.DataFrame()
    df['speech'] = audio_list
    df['labels'] = [re.sub(r'\d+\.mp3$', '', audio) for audio in audio_list]
    print(f"The first five rows of the data {df.head()}")
    # Filter based on the accent
    filtered_df = df[df['labels'].isin(accent_list)]
    # Plot the data distribution
    plt.figure(figsize=(12, 5))
    ax = sns.countplot(data=filtered_df, x='labels', order=filtered_df['labels'].value_counts().index)
    total = len(filtered_df['labels'])
    # Include number of each label on the plot
    for p in ax.patches:
        height = p.get_height()
        count = int(height) 
        percentage = (count / total) * 100 
        ax.annotate(f'{count} ({percentage:.1f}%)',
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    plt.title("Label Distribution of selected accents")
    plt.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.show()

    # Get all speech into list format for easy conversion into Pytorch Dataset format
    audio_path_list = [os.path.join(audio_path, filename) for filename in filtered_df["speech"]]
    labels_list = list(filtered_df["labels"])
    print(f"\nThere are {len(audio_path_list)} audio files with corresponding {len(labels_list)} labels")
    return audio_path_list, labels_list













