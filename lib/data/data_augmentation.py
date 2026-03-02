import numpy as np
from torchvision import transforms
import sys
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import BeeDataset
from train_val_split import train_val_split
from lib.data.preprocessing import TorchPreprocessor


def data_augmented_loader() :
    train_preprocessor = TorchPreprocessor(
        mean=[0.54151865, 0.50277623, 0.33710416],
        std=[0.25970188, 0.24740465, 0.26307435],
        normalize=True,
        augmentation=True,  # On active l'augmentation pour le train
        resize_method="pad",
        target_size=(224, 224)
    )
    val_preprocessor = TorchPreprocessor(
        mean=[0.54151865, 0.50277623, 0.33710416],
        std=[0.25970188, 0.24740465, 0.26307435],
        normalize=True,
        augmentation=False, # Pas d'augmentation pour la validation
        resize_method="pad",
        target_size=(224, 224)
    )

    # C. On crée les datasets
    train_dataset, val_dataset = train_val_split(
        train_transform=train_preprocessor, 
        val_transform=val_preprocessor
    )

    print(f"Train prêt : {len(train_dataset)} images (avec augmentation)")
    print(f"Val prête  : {len(val_dataset)} images (sans augmentation)")

    # ==============================================================================
    # 4. GESTION DU DÉSÉQUILIBRE (WEIGHTED RANDOM SAMPLER)
    # ==============================================================================
    # On récupère uniquement les labels du set d'entraînement
    labels_train = [sample[1] for sample in train_dataset.samples]

    # On compte combien de fois chaque classe apparaît dans l'entraînement
    compte_classes = np.bincount(labels_train)

    # On calcule le poids de chaque classe (inversement proportionnel à sa fréquence)
    # On ajoute un petit epsilon (1e-8) pour éviter la division par zéro si une classe a disparu
    poids_classes = 1.0 / (compte_classes + 1e-8)

    # On attribue à CHAQUE image du train_dataset le poids correspondant à sa classe
    poids_echantillons = [poids_classes[label] for label in labels_train]

    # Création du Sampler : Il va piocher avec remise (replacement=True)
    sampler = WeightedRandomSampler(
        weights=poids_echantillons,
        num_samples=len(poids_echantillons), # Le modèle verra autant d'images par époque qu'avant
        replacement=True
    )


    # ==============================================================================
    # 5. CRÉATION DES DATALOADERS FINAUX
    # ==============================================================================
    # DataLoader d'entraînement (avec le Sampler pour rééquilibrer)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        sampler=sampler, # Pas de 'shuffle=True' ici, le sampler mélange déjà intelligemment
        num_workers=2    # À adapter selon ton processeur (ex: 4 ou 8)
    )

    # DataLoader de validation (Classique, séquentiel)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=2
    )

    return train_loader, val_loader