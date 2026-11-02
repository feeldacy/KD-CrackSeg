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


class DCMAImageDataset(Dataset):
    """Full-resolution crack images for DCMA-Net.

    DCMA-Net consumes one whole image and derives its own 12 local crops
    internally, so this dataset only has to deliver the untouched image and
    mask at the network's native size (544 x 384).

    Args:
        pairs: list of (image_path, mask_path) tuples.
        mask_threshold: pixel value above which a mask pixel counts as crack.
        img_size: target (width, height). DCMA-Net expects (544, 384).
        train: enable random flipping augmentation.
    """

    def __init__(self, pairs, mask_threshold=127, img_size=(544, 384), train=False):
        self.pairs = pairs
        self.thr = mask_threshold / 255.0
        self.img_size = img_size
        self.train = train

        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(self.img_size, Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)

        # joint augmentation: the same flip for image and mask
        if self.train:
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() < 0.5:
                image, mask = TF.vflip(image), TF.vflip(mask)

        image = self.norm(TF.to_tensor(image))          # (3, 384, 544)
        mask = (TF.to_tensor(mask) > self.thr).float()  # (1, 384, 544)

        return image, mask