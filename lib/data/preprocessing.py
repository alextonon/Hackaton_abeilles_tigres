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
    def __init__(self, normalize=True, resize_method="crop", target_size=(224, 224)):
        
        self.mean = [0.54151865, 0.50277623, 0.33710416]
        self.std = [0.25970188, 0.24740465, 0.26307435]
        
        transform_list = []

        # 🔹 Resize strategies
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

        # 🔹 ToTensor (toujours UNE seule fois)
        transform_list.append(transforms.ToTensor())

        # 🔹 Normalization
        if normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )

        self.transform = transforms.Compose(transform_list)

    def __call__(self, img):
        return self.transform(img)