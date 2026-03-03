import torch
from torchvision import transforms
import torchvision.transforms.functional as F

class PadToSquare:
    """
    Classe pour rendre les images carrées ou quoi la team
    """

    def __init__(self, target_size, fill=255):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        max_side = max(w, h)

        # Padding pour rendre carré
        pad_left = (max_side - w) // 2
        pad_top = (max_side - h) // 2
        pad_right = max_side - w - pad_left
        pad_bottom = max_side - h - pad_top

        img = F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)

        # Resize final
        img = F.resize(img, self.target_size)

        return img

class TorchPreprocessor:
    """
    Préparation du dataset préalable à son passage dans un réseau de neurones
    - redimensionnement des images
    - augmentation des données
    """
    
    def __init__(self, 
                 mean=None, 
                 std=None, 
                 normalize=True,
                 augmentation="none",
                 resize_method="crop", 
                 interpolation_method="BILINEAR",
                 resize_value=256,
                 target_size=(224, 224)):
        
        # On initialise la moyenne et l'écart-type de notre jeu de données
        self.mean = mean if mean is not None else [0.54151865, 0.50277623, 0.33710416]
        self.std = std if std is not None else [0.25970188, 0.24740465, 0.26307435]
        
        transform_list = []

        # Initialisation de la méthode d'interpolation
        if interpolation_method == "BILINEAR":
            interpolation = transforms.InterpolationMode.BILINEAR
        elif interpolation_method == "BICUBIC":
            interpolation = transforms.InterpolationMode.BICUBIC
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation_method}")

        # Stratégies de redimensionnement
        if resize_method == "resize":
            transform_list.append(
                transforms.Resize(target_size, interpolation=interpolation)
            )

        elif resize_method == "crop":
            transform_list.append(
                transforms.Resize(resize_value, interpolation=interpolation)
            )
            transform_list.append(
                transforms.CenterCrop(target_size)
            )

        elif resize_method == "pad":
            transform_list.append(
                PadToSquare(target_size)
            )
        

        # Stratégies de data augmentation
        # Light = légères modifications pour les classes courantes d'abeilles
        if augmentation == "light":
            transform_list.extend([
                transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandAugment(num_ops=1, magnitude=5)
            ])

        # Heavy = modifications conséquentes pour les classes rares d'abeilles
        elif augmentation == "heavy":
            transform_list.extend([
                transforms.RandomResizedCrop(target_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                # RandAugment Agressif
                transforms.RandAugment(num_ops=3, magnitude=15)
            ])
        elif augmentation == "RandAugment":
            transform_list.extend([
                transforms.RandomResizedCrop(target_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandAugment(num_ops=2, magnitude=9)
            ])

        # ToTensor, scale le PIL de [0,255] à [0,1]
        transform_list.append(transforms.ToTensor())

        # Normalisation
        if normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )

        self.transform = transforms.Compose(transform_list)

    def __call__(self, img):
        return self.transform(img)