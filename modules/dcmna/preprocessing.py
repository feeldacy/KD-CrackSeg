import torch

def split_image(img, n_w=4, n_h=3):
    """
    img: tensor of shape (B, C, H, W) = (B, C, 384, 544)
    returns: tensor of shape (B, n_h*n_w, C, H/n_h, W/n_w) = (B, 12, C, 128, 136)
    """
    B, C, H, W = img.shape
    ph, pw = H // n_h, W // n_w
    
    patches = img.unfold(2, ph, ph).unfold(3, pw, pw)
    patches = patches.premute(0, 2, 3, 1, 4, 5)
    patches = patches.reshape(B, n_h * n_w, C, ph, pw)
    return patches

def merge_patches(patches, n_w=4, n_h=3):
    """Inverse of split_image: (B, 12, C, 128, 136) -> (B, C, 384, 544)"""
    B, N, C, ph, pw = patches.shape
    x = patches.reshape(B, n_h, n_w, C, ph, pw)
    x = x.permute(0, 3, 1, 4, 2, 5)                      # (B, C, n_h, ph, n_w, pw)
    return x.reshape(B, C, n_h * ph, n_w * pw)    
    