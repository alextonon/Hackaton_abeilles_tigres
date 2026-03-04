import numpy as np
from torchvision import transforms
import sys
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F

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
    

class MixupCollate:
    """
    Intercepte un batch d'images et applique l'algorithme Mixup.
    Fusionne les images et transforme les labels entiers en probabilités (soft labels).
    """
    def __init__(self, num_classes, alpha=0.2):
        self.num_classes = num_classes
        self.alpha = alpha # Contrôle la force du mélange (0.2 est le standard de l'industrie)

    def __call__(self, batch):
        # 1. On laisse PyTorch assembler le batch normalement (32 images, 32 labels)
        images, labels = default_collate(batch)

        # 2. On tire au sort le ratio de mélange (lambda) selon une distribution Beta
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # 3. On crée un ordre aléatoire pour mélanger le batch avec lui-même
        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        # 4. Fusion des images : image_A * 0.8 + image_B * 0.2
        mixed_images = lam * images + (1 - lam) * images[index, :]

        # 5. Fusion des étiquettes (Nécessite de passer en One-Hot Encoding)
        # Ex: L'abeille A (classe 2) devient [0, 0, 1, 0...]
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # Le label devient une probabilité : 80% classe 2, 20% classe 5
        mixed_labels = lam * labels_one_hot + (1 - lam) * labels_one_hot[index, :]

        return mixed_images, mixed_labels


def data_augmented_loader(mean, std, target_size, batch_size=32,
                        apply_augmentation=False,
                        distinguish_classes=False,
                        use_mixup=False,
                        num_classes=50,
                        train_preprocessor_light=None, 
                        train_preprocessor_heavy=None, 
                        train_preprocessor_uniform=None,
                        val_preprocessor=None):
    """
    Loader complet rendant un dataset d'entraînement et un de validation.
    Si apply_augmentation=False : N'applique aucune Data Augmentation.
    Si distinguish_classes=True : Applique light aux communes et heavy aux rares.
    Si distinguish_classes=False : Applique une augmentation conséquente à tout le monde.
    """
    
    # Initialisation des préprocesseurs, avec le choix des méthodes de data augmentation,
    # de la méthode de redimensionnement et de la taille finale des images

    if val_preprocessor is None:    
        val_preprocessor = TorchPreprocessor(
            mean=mean, std=std, normalize=True,
            augmentation="none", 
            resize_method="pad", target_size=target_size
        )

    if not apply_augmentation :

        if train_preprocessor_uniform is None:
            train_preprocessor_uniform = TorchPreprocessor(
                mean=mean, std=std, normalize=True,
                augmentation="none",
                resize_method="pad", target_size=target_size
            )
        
        # On passe directement le preprocessor uniforme au split
        train_dataset, val_dataset = train_val_split(
            train_transform=train_preprocessor_uniform, 
            val_transform=val_preprocessor
        )
        
        train_dataset_final = train_dataset
        train_sampler = None 
        train_shuffle = True

        print(f"Train prêt : {len(train_dataset_final)} images (Pas d'augmentation)")


    else :

        # Cas où l'on augmente de la même manière les images provenant des classes rares
        # et celles provenant des classes communes
        if not distinguish_classes:

            if train_preprocessor_uniform is None:
                train_preprocessor_uniform = TorchPreprocessor(
                    mean=mean, std=std, normalize=True,
                    augmentation="RandAugment",
                    resize_method="pad", target_size=target_size
                )
            
            # On passe directement le preprocessor uniforme au split
            train_dataset, val_dataset = train_val_split(
                train_transform=train_preprocessor_uniform, 
                val_transform=val_preprocessor
            )
            
            train_dataset_final = train_dataset
            train_sampler = None 
            train_shuffle = True

            print(f"Train prêt : {len(train_dataset_final)} images (Augmentation UNIFORME, Tirage CLASSIQUE)")


        # Cas où l'on augmente différemment les images provenant des classes rares
        # et celles provenant des classes communes
        else:
            # Un préprocesseur pour les images que l'on va faiblement augmenter (classes communes)
            if train_preprocessor_light is None:
                train_preprocessor_light = TorchPreprocessor(
                    mean=mean, std=std, normalize=True,
                    augmentation="light", 
                    resize_method="pad", target_size=target_size
                )
            
            # Un préprocesseur pour les images que l'on va fortement augmenter (classes rares)
            if train_preprocessor_heavy is None:
                train_preprocessor_heavy = TorchPreprocessor(
                    mean=mean, std=std, normalize=True,
                    augmentation="heavy", 
                    resize_method="pad", target_size=target_size
                )

            # On fait le split sans transform pour le train (le Wrapper va s'en charger)
            train_dataset, val_dataset = train_val_split(
                train_transform=None, 
                val_transform=val_preprocessor
            )

            labels_train = [sample[1] for sample in train_dataset.samples]
            compte_classes = np.bincount(labels_train)

            # 1. Ciblage des augmentations
            SEUIL_RARE = 100
            dict_rares = {}
            for classe_idx, compte in enumerate(compte_classes):
                if compte < SEUIL_RARE:
                    dict_rares[classe_idx] = train_preprocessor_heavy

            train_dataset_final = TargetedAugmentation(
                dataset_base=train_dataset,
                transform_defaut=train_preprocessor_light,
                dict_transforms_rares=dict_rares
            )

            # 2. Création du Sampler
            poids_classes = 1.0 / (compte_classes + 1e-8)
            poids_echantillons = [poids_classes[label] for label in labels_train]

            train_sampler = WeightedRandomSampler(
                weights=poids_echantillons,
                num_samples=len(poids_echantillons), 
                replacement=True
            )
            train_shuffle = False

            print(f"Train prêt : {len(train_dataset_final)} images (Augmentation CIBLÉE, Weighted SAMPLER)")

    print(f"Val prête  : {len(val_dataset)} images (sans augmentation)")


    train_collate_fn = None
    
    if use_mixup:
        if num_classes is None:
            raise ValueError("Tu dois fournir 'num_classes' si tu actives le Mixup !")
        # On active notre intercepteur de batch
        train_collate_fn = MixupCollate(num_classes=num_classes, alpha=0.2)
        print("-> Mixup : ACTIVÉ (Les labels sont maintenant des probabilités, il faut adapter la loss en conséquence mon coquin )")


    # Création des DataLoader finaux
    train_loader = DataLoader(
        train_dataset_final, 
        batch_size=batch_size, 
        sampler=train_sampler,
        shuffle=train_shuffle,
        collate_fn=train_collate_fn,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )

    return train_loader, val_loader


def final_training_data_loader(mean, std, target_size, batch_size=32,
                        apply_augmentation=False,
                        distinguish_classes=False,
                        use_mixup=False,
                        num_classes=50,
                        train_preprocessor_light=None, 
                        train_preprocessor_heavy=None, 
                        train_preprocessor_uniform=None):
    
    # 1. On charge le dataset global UNE SEULE FOIS pour tout le monde
    full_dataset = BeeDataset(train=True)

    # CAS 1 : AUCUNE AUGMENTATION
    if not apply_augmentation:
        # On force la création d'un préprocesseur vierge pour garantir l'absence d'augmentation
        preprocessor_none = TorchPreprocessor(
            mean=mean, std=std, normalize=True,
            augmentation="none",
            resize_method="pad", target_size=target_size
        )
        
        full_dataset.transform = preprocessor_none
        
        train_dataset_final = full_dataset
        train_sampler = None 
        train_shuffle = True

        print(f"Train prêt : {len(train_dataset_final)} images (Pas d'augmentation)")

    else:
        # CAS 2A : AUGMENTATION UNIFORME
        if not distinguish_classes:
            if train_preprocessor_uniform is None:
                train_preprocessor_uniform = TorchPreprocessor(
                    mean=mean, std=std, normalize=True,
                    augmentation="RandAugment",
                    resize_method="pad", target_size=target_size
                )

            full_dataset.transform = train_preprocessor_uniform
            
            train_dataset_final = full_dataset
            train_sampler = None 
            train_shuffle = True

            print(f"Train prêt : {len(train_dataset_final)} images (Augmentation UNIFORME, Tirage CLASSIQUE)")

        # CAS 2B : AUGMENTATION CIBLÉE + RÉÉQUILIBRAGE
        else:
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

            # On s'assure que le dataset global n'a pas de transformation propre
            full_dataset.transform = None

            labels_train = [sample[1] for sample in full_dataset.samples]
            compte_classes = np.bincount(labels_train)

            # Ciblage
            SEUIL_RARE = 100
            dict_rares = {}
            for classe_idx, compte in enumerate(compte_classes):
                if compte < SEUIL_RARE:
                    dict_rares[classe_idx] = train_preprocessor_heavy

            # Application du Wrapper (TargetedAugmentation) sur le dataset global
            train_dataset_final = TargetedAugmentation(
                dataset_base=full_dataset,
                transform_defaut=train_preprocessor_light,
                dict_transforms_rares=dict_rares
            )

            # Création du Sampler basé sur toutes les données
            poids_classes = 1.0 / (compte_classes + 1e-8)
            poids_echantillons = [poids_classes[label] for label in labels_train]

            train_sampler = WeightedRandomSampler(
                weights=poids_echantillons,
                num_samples=len(poids_echantillons), 
                replacement=True
            )
            train_shuffle = False

            print(f"Train prêt : {len(train_dataset_final)} images (Augmentation CIBLÉE, Weighted SAMPLER)")


    # GESTION DU MIXUP
    train_collate_fn = None
    
    if use_mixup:
        if num_classes is None:
            raise ValueError("Tu dois fournir 'num_classes' si tu actives le Mixup !")
        train_collate_fn = MixupCollate(num_classes=num_classes, alpha=0.2)
        print("-> Mixup : ACTIVÉ (Les labels sont maintenant des probabilités, il faut adapter la loss en conséquence mon coquin)")


    # DATALOADER FINAL
    train_loader = DataLoader(
        train_dataset_final, 
        batch_size=batch_size, 
        sampler=train_sampler,
        shuffle=train_shuffle,
        collate_fn=train_collate_fn,
        num_workers=2
    )

    return train_loader
