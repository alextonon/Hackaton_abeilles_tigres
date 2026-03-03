import sys
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler
import numpy as np
import copy

from .dataset import BeeDataset
from sklearn.model_selection import train_test_split
from collections import Counter


def train_val_split(train_transform=None, val_transform=None):
    """
    Fonction pour extraire un ensemble de validation des données d'entraînement globales,
    en gardant des images de chaque classe à la fois dans l'ensemble d'entraînement et 
    dans celui de validation
    """

    # On charge toutes les données d'entraînement à l'aide de la classe BeeDataset
    full_dataset = BeeDataset(train=True)
    indices = np.arange(len(full_dataset))
    labels = np.array([sample[1] for sample in full_dataset.samples])

    idx_train = []
    idx_val = []

    # On fait le split classe par classe pour garder le contrôle total
    for classe_idx in np.unique(labels):
        # On récupère tous les indices des images de la classe en question
        indices_cette_classe = indices[labels == classe_idx]
        
        # S'il y a au moins 2 images d'abeilles appartenant à cette classe :
        if len(indices_cette_classe) >= 2:
            # On split cette classe spécifiquement, en gardant 20 % des données dans 
            # l'ensemble de validation (ou au moins 1 image)
            t_idx, v_idx = train_test_split(
                indices_cette_classe, 
                test_size=max(1, int(len(indices_cette_classe) * 0.2)),
                random_state=42
            )
            idx_train.extend(t_idx)
            idx_val.extend(v_idx)
        else:
            # Cas critique : 1 seule image au total pour l'espèce
            # On est obligé de la mettre en train, sinon le modèle ne la verra jamais
            # Tant pis pour l'ensemble de validation ce gros boloss
            idx_train.extend(indices_cette_classe)
            print(f"⚠️ Classe {classe_idx} : 1 seule image. Mise en 'train' uniquement.")

    # On copie le dataset entier pour ensuite le filtrer sur les indices concernés
    train_dataset = copy.deepcopy(full_dataset)
    val_dataset = copy.deepcopy(full_dataset)

    # On filtre le dataset entier pour ne garder que les indices correspondant à l'entraînement,
    # d'une part, et à la validation d'autre part
    train_dataset.samples = [full_dataset.samples[i] for i in idx_train]
    val_dataset.samples = [full_dataset.samples[i] for i in idx_val]

    # On initialise le paramètre transform des datasets si data augmentation il y a
    if train_transform is not None:
        train_dataset.transform = train_transform
    if val_transform is not None:
        val_dataset.transform = val_transform

    return train_dataset, val_dataset