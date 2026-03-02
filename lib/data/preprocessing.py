import torch
from torchvision import transforms
import torchvision.transforms.functional as F

class PadToSquare:
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
    def __init__(self, 
                 mean=None, 
                 std=None, 
                 normalize=True,
                 augmentation="none",
                 resize_method="crop", 
                 target_size=(224, 224)):
        
        self.mean = mean if mean is not None else [0.54151865, 0.50277623, 0.33710416]
        self.std = std if std is not None else [0.25970188, 0.24740465, 0.26307435]
        
        transform_list = []

        # Resize strategies
        if resize_method == "resize":
            transform_list.append(
                transforms.Resize(target_size)
            )

        elif resize_method == "crop":
            transform_list.append(
                transforms.Resize(target_size)
            )
            transform_list.append(
                transforms.CenterCrop(target_size)
            )

        elif resize_method == "pad":
            transform_list.append(
                PadToSquare(target_size)
            )
        
        if augmentation == "light":
            transform_list.extend([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2),
            ])
        elif augmentation == "heavy":
            transform_list.extend([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=45),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            ])
        elif augmentation == "RandAugment":
            transform_list.append(
                transforms.RandAugment(num_ops=2, magnitude=9)
            )

        # ToTensor, scale le PIL de [0,255] à [0,1]
        transform_list.append(transforms.ToTensor())

        # Normalization
        if normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )

        self.transform = transforms.Compose(transform_list)

    def __call__(self, img):
        return self.transform(img)