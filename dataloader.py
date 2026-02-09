from PIL import Image, UnidentifiedImageError
import os
import PIL
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from dataset import download_and_prepare_dataset

def collate_fn(items: list[dict]) -> dict:
    items = [item for item in items if item is not None]
    if len(items) == 0:
        return None
    
    images = np.stack([item["image"] for item in items], axis=0)
    labels = np.array([item["label"] for item in items], dtype=np.int64)

    return {
        "images": torch.tensor(images, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

class CatDogDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        super().__init__()
        self.transform = transform
        self.data = []

        search_paths = os.path.join(data_dir, 'PetImages')
        if not os.path.exists(search_paths):
            search_paths = data_dir
        
        categories = {'Cat': 0, 'Dog': 1}

        for category, label in categories.items():
            categories_path = os.path.join(search_paths, category)
            if os.path.exists(categories_path):
                for filename in os.listdir(categories_path):
                    self.data.append(
                        {
                            "path": os.path.join(categories_path, filename),
                            "label": label
                        }
                    )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int) -> dict:
        item = self.data[index]
        path = item["path"]
        label = item["label"]

        try:
            image = PIL.Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            image = image.numpy()
            return {
                "image": image,
                "label": label
            }
        except Exception as e:
            return None

def get_dataloaders(data_dir, image_size, batch_size, use_augmentation=False):
    # Download and prepare dataset
    download_and_prepare_dataset(data_dir)
    # Define transforms
    base_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = base_transform
    
    dataset = CatDogDataset(data_dir, transform=train_transform)
    # Split dataset
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    generator = torch.Generator().manual_seed(42)

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    class TransformedSubset(Dataset):
        def __init__(self, subset, transform):
            super().__init__()
            self.subset = subset
            self.transform = transform
        
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, index):
            item = self.subset[index]

            dataset_ref = self.subset.dataset
            real_index = self.subset.indices[index]

            original_item = dataset_ref.data[real_index]
            path = original_item["path"]
            label = original_item["label"]

            try:
                image = Image.open(path).convert("RGB")
                image = self.transform(image)
                return {
                    "image": image.numpy(),
                    "label": label
                }
            except Exception as e:
                return None
        
    train_set = TransformedSubset(train_set, train_transform)
    val_set = TransformedSubset(val_set, base_transform)
    test_set = TransformedSubset(test_set, base_transform)

    return train_set, val_set, test_set



 