import argparse, random, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from data.registry import get_pairs, split_pairs
from dataset import SegDataset
from teacher_lactnet import LACTNet
from modules.total_loss import build_loss

# ---------- 1. setup ----------
p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--data_root", default=None)
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--bs", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--out", default="runs/exp1")
p.add_argument("--loss", default="dice_bce", choices=["dice_bce", "dice_focal"])
p.add_argument("--bnd_loss", default="dice_bce", choices=["dice_bce", "dice_focal"])
p.add_argument("--bnd_w", type=float, default=0.5)
p.add_argument("--aux_w", type=float, default=0.5)
p.add_argument("--val_ratio", type=float, default=0.1)   # used only if the dataset has no val split
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 2. data ----------
train_pairs, thr = get_pairs(args.dataset, "train", data_root=args.data_root)

try:
    val_pairs, _ = get_pairs(args.dataset, "val", data_root=args.data_root)   # e.g. Crack500
    print("Using the dataset's own val split")
except KeyError:                                                              # e.g. DeepCrack
    train_pairs, val_pairs = split_pairs(train_pairs, args.val_ratio, args.seed)
    print(f"No val split, held out {len(val_pairs)} of train for validation")

# record which files were used for validation (reproducibility)
with open(f"{args.out}/val_files.txt", "w") as f:
    f.write("\n".join(str(pr[0].name) for pr in val_pairs))

train_loader = DataLoader(SegDataset(train_pairs, thr, 256, train=True),
                          batch_size=args.bs, shuffle=True, num_workers=8,
                          pin_memory=True, drop_last=True)
val_loader = DataLoader(SegDataset(val_pairs, thr, 256, train=False),
                        batch_size=args.bs, shuffle=False, num_workers=4)

# Evaluate the held-out test split once training is complete:
# python evaluate.py --dataset DATASET --checkpoint runs/exp1/best.pth

# ---------- 3. model, loss, optimizer ----------
model = LACTNet().to(device)

sem_criterion = build_loss(args.loss)
bnd_criterion = build_loss(args.bnd_loss)

def make_boundary(mask, k=3):
    """boundary GT from the mask: dilation - erosion"""
    dil = F.max_pool2d(mask, k, 1, k // 2)
    ero = -F.max_pool2d(-mask, k, 1, k // 2)
    return (dil - ero).clamp(0, 1)

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

# ---------- 4. validation ----------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    tp = fp = fn = 0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        pred = torch.sigmoid(model(img)["pred_semantic"]) > 0.5
        m = mask > 0.5
        tp += (pred & m).sum().item()
        fp += (pred & ~m).sum().item()
        fn += (~pred & m).sum().item()
    prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
    return dict(iou=tp / (tp + fp + fn + 1e-8),
                f1=2 * prec * rec / (prec + rec + 1e-8),
                precision=prec, recall=rec)

# ---------- 5. training loop ----------
best_f1 = 0.0
for epoch in range(args.epochs):
    model.train()
    running = 0.0
    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)
        bnd = make_boundary(mask)

        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            out = model(img, gts=[mask, mask])
            loss = (sem_criterion(out["pred_semantic"], mask)
                    + args.bnd_w * bnd_criterion(out["pred_boundary"], bnd)
                    + args.aux_w * out["supervision_loss"])

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        running += loss.item()

    sched.step()
    m = evaluate(val_loader)
    print(f"epoch {epoch+1}/{args.epochs} loss {running/len(train_loader):.4f} | "
          f"val IoU {m['iou']:.4f} F1 {m['f1']:.4f} P {m['precision']:.4f} R {m['recall']:.4f}")

    # ---------- 6. checkpoints ----------
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "epoch": epoch}, f"{args.out}/last.pth")
    if m["f1"] > best_f1:
        best_f1 = m["f1"]
        torch.save(model.state_dict(), f"{args.out}/best.pth")
        print(f"  saved best.pth (val F1 {best_f1:.4f})")