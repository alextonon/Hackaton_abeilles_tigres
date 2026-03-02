import sys
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler
import numpy as np
import copy

from dataset import BeeDataset
from sklearn.model_selection import train_test_split
from collections import Counter

def train_val_split(root_dir) : 

    # 1. On instancie ta classe (pas besoin de 'transform' ici, on veut juste lire les chemins)
    # Remplace 'data/train' par le chemin exact de ton dossier
    full_dataset = BeeDataset(root_dir=root_dir)

    indices = np.arange(len(full_dataset))
    labels = np.array([sample[1] for sample in full_dataset.samples])

    idx_train = []
    idx_val = []

    # On fait le split classe par classe pour garder le contrôle total
    for classe_idx in np.unique(labels):
        # On récupère tous les indices des images de CETTE classe
        indices_cette_classe = indices[labels == classe_idx]
        
        if len(indices_cette_classe) >= 2:
            # On split cette classe spécifiquement
            # test_size=0.2 ou au moins 1 image
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
            idx_train.extend(indices_cette_classe)
            print(f"⚠️ Classe {classe_idx} : 1 seule image. Mise en 'train' uniquement.")

    train_dataset = copy.deepcopy(full_dataset)
    val_dataset = copy.deepcopy(full_dataset)

    # 4. On écrase leur liste d'échantillons pour ne garder QUE ceux sélectionnés
    train_dataset.samples = [full_dataset.samples[i] for i in idx_train]
    val_dataset.samples = [full_dataset.samples[i] for i in idx_val]

    return train_dataset, val_dataset