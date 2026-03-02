import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image

class BeeDatasetOld(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir:
            data/
                train/
                    classe_1/
                        img1.jpg
                        img2.jpg
                    classe_2/
                        img3.jpg
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.samples = []
        self.class_to_idx = {}

        # 🔹 Récupérer les classes
        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        # 🔹 Indexer toutes les images
        for cls_name in classes:
            class_folder = os.path.join(root_dir, cls_name)
            
            if not os.path.isdir(class_folder):
                continue
                
            for file_name in os.listdir(class_folder):
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(class_folder, file_name)
                    label = self.class_to_idx[cls_name]
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        with Image.open(path).convert("RGB") as img:
            if self.transform:
                img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

class BeeDataset(Dataset):
    def __init__(self, train, transform=None):
        self.train_csv_dir = "data/train.csv"
        self.test_csv_dir = "data/test.csv"
        
        self.train = train

        self.transform = transform

        if train:
            file = pd.read_csv(self.train_csv_dir, sep=",")
            image_paths = file["id"]
            label = file["label"]
            image_paths = [os.path.join("data/", img) for img in image_paths]
            self.samples = list(zip(image_paths, label))

        else:
            file = pd.read_csv(self.test_csv_dir, sep=",")
            id = file["id"]
            image_path = file["image"]
            image_paths = [os.path.join("data/test", img) for img in image_path]
            self.samples = list(zip(image_paths, id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
            path, extra = self.samples[idx] # extra = label (train) ou id (test)

            img = Image.open(path).convert("RGB")
            
            if self.transform:
                img = self.transform(img)

            if self.train:
                return img, torch.tensor(extra, dtype=torch.long)
            else:
                return img, extra