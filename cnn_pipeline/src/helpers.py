import os
import requests
from zipfile import ZipFile
from pathlib import Path

def download_and_extract_data(url, directory):
    try:
        # Check if the directory exists and is empty
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Only proceed if the directory is empty
        if not os.listdir(directory):
            # Download the url data as `data.zip` in `directory`
            response = requests.get(url, stream= True)
            zip_path = os.path.join(directory, 'data.zip')
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size= 128):
                    f.write(chunk)
            # Extract `data.zip` and remove it afterwards:
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(directory)
            os.remove(zip_path)

            print(f"Data downloaded and extracted to '{directory}'.")
        else:
            print(
                f"Directory '{directory}' already exists and is not empty."
                "To re-download, please remove the directory first."
            )
    except OSError as e:
        print(f"An error occurred: {e}")
