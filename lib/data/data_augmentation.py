import numpy as np
from torchvision import transforms
import sys
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset

from .dataset import BeeDataset
from .train_val_split import train_val_split
from .preprocessing import TorchPreprocessor


class TargetedAugmentation(Dataset):
    """
    Wrapper qui gère l'augmentation de données en distinguant si l'image concernée
    appartient à une classe rare ou non
    Lorsque le DataLoader pioche une image, on fait appel à cette classe pour qu'elle nous dise
    si l'image appartient à une classe rare, auquel cas on programme la data augmentation
    en conséquence
    """

    def __init__(self, dataset_base, transform_defaut, dict_transforms_rares):
        self.dataset_base = dataset_base
        # On s'assure que le dataset de base n'applique rien de lui-même
        self.dataset_base.transform = None 
        
        self.transform_defaut = transform_defaut
        self.dict_transforms_rares = dict_transforms_rares

    def __len__(self):
        return len(self.dataset_base)

    def __getitem__(self, idx):
        # On lit le chemin et le label (la base renvoie l'image non transformée, ex: PIL)
        image, label_tensor = self.dataset_base[idx]
        label_idx = label_tensor.item()
        
        # On applique la transformation spécifique si la classe est ciblée
        if label_idx in self.dict_transforms_rares:
            image = self.dict_transforms_rares[label_idx](image)
        # Sinon, la transformation d'entraînement par défaut (light)
        else:
            image = self.transform_defaut(image)
            
        return image, label_tensor
    

def data_augmented_loader(mean, std, target_size, batch_size=32, train_preprocessor_light= None, train_preprocessor_heavy=None, val_preprocessor=None):
    """
    Loader complet rendant un dataset d'entraînement et un de validation
    """
    
    # Initialisation des préprocesseurs, avec le choix des méthodes de data augmentation,
    # de la méthode de redimensionnement et de la taille finale des images

    if train_preprocessor_light is None:
        train_preprocessor_light = TorchPreprocessor(
            mean=mean, std=std, normalize=True,
            augmentation="light", 
            resize_method="pad", target_size=target_size
        )
    
    if train_preprocessor_heavy is None:
        train_preprocessor_heavy = TorchPreprocessor(
        mean=mean, std=std, normalize=True,
        augmentation="heavy", 
        resize_method="pad", target_size=target_size
    )
    if val_preprocessor is None:    
        val_preprocessor = TorchPreprocessor(
            mean=mean, std=std, normalize=True,
            augmentation="none", 
            resize_method="pad", target_size=target_size
        )

    # Séparation des datasets
    # On passe val_preprocessor pour la validation.
    # Pour le train, on ne passe rien pour l'instant (None), car le Wrapper va s'en charger.
    train_dataset, val_dataset = train_val_split(
        train_transform=None, 
        val_transform=val_preprocessor
    )

    # Ciblage des classes rares (Class-Aware Augmentation)
    labels_train = [sample[1] for sample in train_dataset.samples]
    compte_classes = np.bincount(labels_train)

    # Définir quelles classes sont rares (ici celles ayant moins de 100 images)
    SEUIL_RARE = 100
    dict_rares = {}
    for classe_idx, compte in enumerate(compte_classes):
        if compte < SEUIL_RARE:
            dict_rares[classe_idx] = train_preprocessor_heavy

    # On enrobe le train_dataset avec notre logique conditionnelle
    train_dataset_ciblé = TargetedAugmentation(
        dataset_base=train_dataset,
        transform_defaut=train_preprocessor_light,
        dict_transforms_rares=dict_rares
    )

    print(f"Train prêt : {len(train_dataset_ciblé)} images (avec augmentation ciblée)")
    print(f"Val prête  : {len(val_dataset)} images (sans augmentation)")

    # Gestion du déséquilibre à l'aide d'un Weighted Random Sampler
    # On veut que l'on ait plus de chances de choisir les images appartenant à des
    # classes sous-représentées
    # On incorpore donc un sampler aléatoire, pondéré par la taille des différentes populations
    poids_classes = 1.0 / (compte_classes + 1e-8)
    poids_echantillons = [poids_classes[label] for label in labels_train]

    sampler = WeightedRandomSampler(
        weights=poids_echantillons,
        num_samples=len(poids_echantillons), 
        replacement=True
    )

    # Création des dataloaders finaux
    # Pour la validation on n'utilise pas le sampler
    train_loader = DataLoader(
        train_dataset_ciblé, # <--- On utilise le dataset enrobé ici
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )

    return train_loader, val_loader