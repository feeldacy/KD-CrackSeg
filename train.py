import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.registry import get_pairs, split_pairs
from dataset import SegDataset
from modules.total_loss import build_loss
from teacher_lactnet import LACTNet


def make_boundary(mask, k=3):
    """Boundary GT from the mask: dilation - erosion."""
    dil = F.max_pool2d(mask, k, 1, k // 2)
    ero = -F.max_pool2d(-mask, k, 1, k // 2)
    return (dil - ero).clamp(0, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        pred = torch.sigmoid(model(img)["pred_semantic"]) > 0.5
        m = mask > 0.5
        tp += (pred & m).sum().item()
        fp += (pred & ~m).sum().item()
        fn += (~pred & m).sum().item()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    return dict(
        iou=tp / (tp + fp + fn + 1e-8),
        f1=2 * prec * rec / (prec + rec + 1e-8),
        precision=prec,
        recall=rec,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="runs/exp1")
    parser.add_argument("--loss", default="dice_bce", choices=["dice_bce", "dice_focal"])
    parser.add_argument("--bnd_loss", default="dice_bce", choices=["dice_bce", "dice_focal"])
    parser.add_argument("--bnd_w", type=float, default=0.5)
    parser.add_argument("--aux_w", type=float, default=0.5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_pairs, thr = get_pairs(args.dataset, "train", data_root=args.data_root)
    try:
        val_pairs, _ = get_pairs(args.dataset, "val", data_root=args.data_root)
        print("Using the dataset's own val split")
    except KeyError:
        train_pairs, val_pairs = split_pairs(train_pairs, args.val_ratio, args.seed)
        print(f"No val split, held out {len(val_pairs)} of train for validation")

    with open(f"{args.out}/val_files.txt", "w") as f:
        f.write("\n".join(str(pair[0].name) for pair in val_pairs))

    train_loader = DataLoader(
        SegDataset(train_pairs, thr, 256, train=True),
        batch_size=args.bs,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        SegDataset(val_pairs, thr, 256, train=False),
        batch_size=args.bs,
        shuffle=False,
        num_workers=4,
    )

    # Evaluate the held-out test split once training is complete:
    # python evaluate.py --dataset DATASET --checkpoint runs/exp1/best.pth

    model = LACTNet().to(device)
    sem_criterion = build_loss(args.loss)
    bnd_criterion = build_loss(args.bnd_loss)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    best_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            bnd = make_boundary(mask)

            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                out = model(img, gts=[mask, mask])
                loss = (
                    sem_criterion(out["pred_semantic"], mask)
                    + args.bnd_w * bnd_criterion(out["pred_boundary"], bnd)
                    + args.aux_w * out["supervision_loss"]
                )

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running += loss.item()

        sched.step()
        metrics = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch + 1}/{args.epochs} loss {running / len(train_loader):.4f} | "
            f"val IoU {metrics['iou']:.4f} F1 {metrics['f1']:.4f} "
            f"P {metrics['precision']:.4f} R {metrics['recall']:.4f}"
        )

        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch},
            f"{args.out}/last.pth",
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), f"{args.out}/best.pth")
            print(f"  saved best.pth (val F1 {best_f1:.4f})")


if __name__ == "__main__":
    main()
