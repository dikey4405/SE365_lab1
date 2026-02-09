import os
import torch
import requests
import zipfile
import shutil
from PIL import Image, UnidentifiedImageError

url = "https://download.microsoft.com/download/3/e/1/3e1c3f21-ecdb-4869-8368-6deba77b919f/kagglecatsanddogs_5340.zip"

def download_and_prepare_dataset(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    zip_path = os.path.join(data_dir, "cats_and_dogs.zip")
    extracted_path = os.path.join(data_dir, 'PetImages')

    if not os.path.exists(zip_path):
        if not os.path.exists(extracted_path):
            try:
                response = requests.get(url, stream=True)
                with open(zip_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            file.write(chunk)
            except Exception as e:
                print(f"Error downloading dataset: {e}")
                return None
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
        except zipfile.BadZipFile as e:
            print(f"Error extracting dataset: {e}")
            return None

    # Clean corrupted images
    corrupted_count = 0
    for root, dirs, files in os.walk(extracted_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()
            except (UnidentifiedImageError, IOError):
                corrupted_count += 1
                os.remove(file_path)
    
    return extracted_path
