import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

class Preprocessor:
    def __init__(self, normalize=True, resize_method="crop", target_size=(224, 224)):
        self.mean = np.array([0.54151865, 0.50277623, 0.33710416])    
        self.std = np.array([0.25970188, 0.24740465, 0.26307435])
        self.normalize_flag = normalize
        self.resize_method = resize_method
        self.target_size = target_size
    
    def normalize(self, img_array):
        return (img_array - self.mean) / self.std
    
    def resize_img(self, img, method, target_size=(224, 224)):
        tw, th = target_size
        
        if method == "resize":
            # Déformation directe
            return img.resize(target_size, Image.Resampling.LANCZOS)
        
        elif method == "crop":
            # Logique : Redimensionner le PETIT côté pour qu'il touche le bord du cadre, puis couper l'excès
            return ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
            
        elif method == "pad":
            # Logique : Redimensionner le GRAND côté pour qu'il rentre dans le cadre, puis ajouter du blanc
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            new_img = Image.new("RGB", target_size, (255, 255, 255))
            
            offset = ((tw - img.size[0]) // 2, (th - img.size[1]) // 2)
            new_img.paste(img, offset)
            return new_img
    
    def __call__(self, img):
        img_resized = self.resize_img(img, method=self.resize_method, target_size=self.target_size)
        
        img_array = np.array(img_resized).astype(np.float32) / 255.0

        if self.normalize_flag:
            img_array = self.normalize(img_array)
        
        img_tensor = img_array.transpose(2, 0, 1) # Convertir de (H, W, C) à (C, H, W) pour torch
        return img_tensor