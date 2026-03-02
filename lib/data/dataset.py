import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class BeeDataset(Dataset):
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