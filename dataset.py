import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


class SegDataset(Dataset):
    def __init__(self, pairs, mask_threshold=127, img_size=256, train=False):
        self.pairs = pairs
        self.thr = mask_threshold / 255.0      # ToTensor scales pixels to [0, 1]
        self.img_size = img_size
        self.train = train

        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])   # ImageNet stats

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        size = (self.img_size, self.img_size)
        image = image.resize(size, Image.BILINEAR)
        mask = mask.resize(size, Image.NEAREST)    # nearest keeps mask values clean

        # simple joint augmentation: same flip for image and mask
        if self.train:
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() < 0.5:
                image, mask = TF.vflip(image), TF.vflip(mask)

        image = self.norm(TF.to_tensor(image))     # (3, H, W)
        mask = (TF.to_tensor(mask) > self.thr).float()   # (1, H, W), values 0 or 1

        return image, mask
