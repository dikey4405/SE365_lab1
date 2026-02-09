import os
import shutil
import requests
import zipfile
from PIL import Image, UnidentifiedImageError

DATASET_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"

def download_and_prepare_dataset(working_dir):
    target_folder = os.path.join(working_dir, "PetImages")
    kaggle_input_root = "/kaggle/input"
    
    if not os.path.exists(target_folder):
        print(f"Data chưa có tại {target_folder}. Đang tìm kiếm nguồn...")
        
        found_in_kaggle = False
        
        if os.path.exists(kaggle_input_root):
            for root, dirs, files in os.walk(kaggle_input_root):
                if "PetImages" in dirs:
                    source_path = os.path.join(root, "PetImages")
                    print(f"--> Tìm thấy dataset tại Kaggle Input: {source_path}")
                    print("--> Đang copy sang /kaggle/working để có quyền ghi/xóa (mất khoảng 1-2 phút)...")
                    shutil.copytree(source_path, target_folder)
                    found_in_kaggle = True
                    break
        
        if not found_in_kaggle:
            print("--> Không tìm thấy trong Kaggle Input, chuyển sang chế độ tải từ Web (Local)...")
            prepare_local_dataset(working_dir)

    print(f"Đang kiểm tra và xóa ảnh lỗi trong {target_folder}...")
    num_removed = 0
    
    if os.path.exists(target_folder):
        for root, dirs, files in os.walk(target_folder):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".db"): # Xóa thumbs.db
                    os.remove(file_path)
                    continue
                try:
                    img = Image.open(file_path)
                    img.verify() 
                except (IOError, SyntaxError, UnidentifiedImageError):
                    os.remove(file_path)
                    num_removed += 1
    
    print(f"Hoàn tất. Đã xóa {num_removed} ảnh lỗi.")
    return target_folder

def prepare_local_dataset(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    zip_path = os.path.join(data_dir, "cats_and_dogs.zip")
    
    if not os.path.exists(zip_path):
        print(f"Downloading dataset...")
        try:
            response = requests.get(DATASET_URL, stream=True)
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            print(f"Download Error: {e}")
            return

    print("Extracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    except zipfile.BadZipFile:
        print("Zip file error.")
