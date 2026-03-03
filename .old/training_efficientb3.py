# %% [markdown]
# # A FAIRE POUR LANCER UN GROS ENTRAINEMENT
# 
# - utiliser un modèle b3 (ou +) que b0
# - en csq changer le target size dans le preprocessor
# - augmenter nombre d'epoch
# - batch size ?
# 
# Optionnel :
# - tester une autre loss

# %% [markdown]
# # **0. Librairies**

# %%
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os


sys.path.append(os.path.abspath(".."))

from lib.data.dataset import BeeDataset

from lib.data.preprocessing import TorchPreprocessor

from lib.data.train_val_split import train_val_split

from lib.data.preprocessing import TorchPreprocessor

from lib.data.data_augmentation import data_augmented_loader

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-4
num_classes = 50

notebook_dir = os.getcwd()

data_dir = os.path.abspath(os.path.join(notebook_dir, "..", "data"))

# %% [markdown]
# ## **1. Preprocessing**

# %%
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# preprocessor = TorchPreprocessor(
#     normalize=True,
#     mean = IMAGENET_MEAN,
#     std = IMAGENET_STD,
#     resize_method="pad",
#     target_size=(224, 224),
# )

# train_dataset, val_dataset = train_val_split(train_transform=preprocessor, val_transform=preprocessor)


# %%
import lib.data.preprocessing as prep
print(prep.__file__)

# %%
heavy_training_preprocessor = TorchPreprocessor(
    normalize=True,
    mean = IMAGENET_MEAN,
    std = IMAGENET_STD,
    augmentation="heavy",
    resize_method="pad",
    target_size=(300, 300),
    interpolation_method="BICUBIC",
)

light_training_preprocessor = TorchPreprocessor(
    normalize=True,
    mean = IMAGENET_MEAN,
    std = IMAGENET_STD,
    augmentation="light",
    resize_method="pad",
    target_size=(300, 300),
    interpolation_method="BICUBIC",
)

val_preprocessor = TorchPreprocessor(
    normalize=True,
    mean = IMAGENET_MEAN,
    std = IMAGENET_STD,
    augmentation="none",
    resize_method="pad",
    target_size=(300, 300),
    interpolation_method="BICUBIC",
)


train_loader, val_loader = data_augmented_loader(IMAGENET_MEAN, IMAGENET_STD, target_size=(300, 300), batch_size=BATCH_SIZE, train_preprocessor_light=light_training_preprocessor, train_preprocessor_heavy=heavy_training_preprocessor, val_preprocessor=val_preprocessor)

# %% [markdown]
# ## **2. Modèle**

# %%
from torch.optim.lr_scheduler import CosineAnnealingLR

def create_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b3(weights="IMAGENET1K_V1") #mettre b3 si ca marche
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )
    return model

model = create_model(num_classes)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

# --- Variante ---
# pip install torchmetrics
# from torchmetrics.classification import MulticlassFocalLoss
# criterion = MulticlassFocalLoss(num_classes=num_classes, alpha=0.25, gamma=2.0)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# %% [markdown]
# ## **3. F1-score**

# %%
import numpy as np

def compute_f1(all_labels, all_preds, num_classes):
    f1_per_class = []

    for cls in range(num_classes):
        # True Positives
        TP = np.sum((np.array(all_preds) == cls) & (np.array(all_labels) == cls))
        # False Positives
        FP = np.sum((np.array(all_preds) == cls) & (np.array(all_labels) != cls))
        # False Negatives
        FN = np.sum((np.array(all_preds) != cls) & (np.array(all_labels) == cls))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0

        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_per_class.append(f1)

    # F1 macro : moyenne des classes
    f1_macro = np.mean(f1_per_class)
    return f1_macro, f1_per_class

# %% [markdown]
# ## **4. Fonctions de training et validation**

# %%
def train_one_epoch():
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for images, labels in tqdm(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 🔹 Calcul F1 avec ta fonction existante
    f1_macro, f1_per_class = compute_f1(all_labels, all_preds, num_classes)

    # 🔹 Affichage
    # print(f"Train F1 macro: {f1_macro:.4f}")

    return total_loss / len(train_loader), correct / total, f1_macro, f1_per_class


# %%
import torch

def validate():
    model.eval()
    total_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calcul F1 manuel par classe
    f1_per_class = []
    for cls in range(num_classes):
        TP = np.sum((np.array(all_preds) == cls) & (np.array(all_labels) == cls))
        FP = np.sum((np.array(all_preds) == cls) & (np.array(all_labels) != cls))
        FN = np.sum((np.array(all_preds) != cls) & (np.array(all_labels) == cls))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_per_class.append(f1)
    
    f1_macro = np.mean(f1_per_class)

    return total_loss / len(val_loader), f1_macro, f1_per_class

# %% [markdown]
# ## **5. Entrainement**

# %% [markdown]
# **Vérif des labels**

# %%
# all_labels = [label for _, label in train_dataset]
# print("Label min:", min(all_labels))
# print("Label max:", max(all_labels))
# print("Nombre de classes uniques:", len(set(all_labels)))

# # Récupérer tous les labels uniques triés
# all_labels = sorted(set(label for _, label in train_dataset.samples))
# label_to_index = {label: i for i, label in enumerate(all_labels)}

# # Remapper les labels dans le dataset
# # for i, (path, label) in enumerate(train_dataset.samples):
# #     train_dataset.samples[i] = (path, label_to_index[label])

# %% [markdown]
# **Entrainement**

# %%
import csv
best_f1 = 0.0
best_model_path = "best_model.pth"

# Configuration du logging CSV
csv_file = "training_log.csv"
fieldnames = ['epoch', 'train_loss', 'train_acc', 'train_f1_macro', 'val_loss', 'val_f1_macro']

# Initialisation du fichier (écrase le précédent s'il existe)
with open(csv_file, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

for epoch in range(EPOCHS):
    train_loss, train_acc, train_f1_macro, train_f1_per_class = train_one_epoch()
    scheduler.step()
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Train F1 Macro: {train_f1_macro:.4f}")
    print(f"Train F1 per class: {train_f1_per_class}")

    val_loss, val_f1_macro, val_f1_per_class = validate()
    print(f"Val   Loss: {val_loss:.4f}")
    print(f"Val   F1 Macro: {val_f1_macro:.4f}")
    print(f"Val   F1 per class: {val_f1_per_class}")

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1_macro': train_f1_macro,
            'val_loss': val_loss,
            'val_f1_macro': val_f1_macro
        })

    # 🔹 Sauvegarde du meilleur modèle
    if val_f1_macro > best_f1:
        best_f1 = val_f1_macro
        torch.save(model.state_dict(), best_model_path)
        print(f" Nouveau meilleur modèle sauvegardé ! F1 Macro = {best_f1:.4f}")

# %% [markdown]
# ## **6. Création du fichier submission**

# %%
from torch.utils.data import DataLoader
import pandas as pd
import torch
from lib.data.dataset import BeeDataset

def submission(model, batch_size=32, transform=None, model_path="best_model.pth", save_path="submission.csv"):
    # Charger le modèle sur le bon device
    model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    model.to(DEVICE)
    model.eval()
    
    # Dataset de test
    test_dataset = BeeDataset(train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_ids = []
    all_preds = []
    
    with torch.no_grad():
        for imgs, ids in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            
            preds = torch.argmax(outputs, dim=1)
            
            # Convertir preds en int et ids en int ou str
            all_preds.extend(preds.cpu().tolist())
            all_ids.extend([int(x) if isinstance(x, torch.Tensor) else x for x in ids])
    
    submission_df = pd.DataFrame({
        "id": all_ids,
        "label": all_preds
    })
    
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")

# %%
preprocessor = TorchPreprocessor(
    normalize=True,
    mean = IMAGENET_MEAN,
    std = IMAGENET_STD,
    resize_method="pad",
    target_size=(224, 224),
)


test_dataset = BeeDataset(train=False, transform=preprocessor)


submission(model, batch_size=32, transform=preprocessor, save_path="submission.csv")


