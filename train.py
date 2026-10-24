import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.registry import get_pairs, split_pairs
from dataset import SegDataset
from modules.total_loss import build_loss
from student_lactnet import LACTNet as StudentLACTNet
from teacher_lactnet import LACTNet as TeacherLACTNet


def make_boundary(mask, k=3):
    """Boundary GT from the mask: dilation - erosion."""
    dil = F.max_pool2d(mask, k, 1, k // 2)
    ero = -F.max_pool2d(-mask, k, 1, k // 2)
    return (dil - ero).clamp(0, 1)


def distillation_loss(student_logits, teacher_logits, temperature):
    """Bernoulli KL distillation for binary segmentation logits."""
    student_logits = student_logits.float() / temperature
    teacher_probs = torch.sigmoid(teacher_logits.float() / temperature)
    teacher_probs = teacher_probs.clamp(1e-7, 1.0 - 1e-7)
    student_log_probs = torch.cat(
        [F.logsigmoid(-student_logits), F.logsigmoid(student_logits)], dim=1
    )
    teacher_probs = torch.cat([1.0 - teacher_probs, teacher_probs], dim=1)
    return (
        F.kl_div(student_log_probs, teacher_probs, reduction="none")
        .sum(dim=1)
        .mean()
        * temperature**2
    )


@torch.no_grad()
def evaluate(model, loader, device, progress_desc="Validation"):
    model.eval()
    tp = fp = fn = 0
    for img, mask in tqdm(loader, desc=progress_desc, leave=False):
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
    parser = argparse.ArgumentParser(
        description="Train LACTNet, its student, or the student with teacher distillation.",
        epilog=(
            "Examples: --mode standard; --mode student; "
            "--mode distill --teacher_checkpoint runs/teacher/best.pth"
        ),
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument(
        "--mode",
        choices=["standard", "distill", "student"],
        default="standard",
        help="standard trains the teacher; student trains LACTNet-S without a teacher; "
        "distill trains LACTNet-S using a frozen teacher",
    )
    parser.add_argument(
        "--teacher_checkpoint",
        type=Path,
        default=None,
        help="Required for --mode distill; accepts best.pth or last.pth",
    )
    parser.add_argument(
        "--kd_w",
        type=float,
        default=1.0,
        help="Weight of teacher-student distillation loss",
    )
    parser.add_argument(
        "--kd_temperature",
        type=float,
        default=2.0,
        help="Temperature for teacher-student distillation",
    )
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

    if args.mode == "distill" and args.teacher_checkpoint is None:
        parser.error("--teacher_checkpoint is required when --mode distill")
    if args.teacher_checkpoint is not None and not args.teacher_checkpoint.is_file():
        parser.error(f"Teacher checkpoint does not exist: {args.teacher_checkpoint}")
    if args.kd_temperature <= 0:
        parser.error("--kd_temperature must be greater than 0")
    if args.kd_w < 0:
        parser.error("--kd_w cannot be negative")

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
        pin_memory=torch.cuda.is_available(),
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

    model_cls = TeacherLACTNet if args.mode == "standard" else StudentLACTNet
    model = model_cls().to(device)
    teacher = None
    if args.mode == "distill":
        teacher = TeacherLACTNet().to(device)
        checkpoint = torch.load(
            args.teacher_checkpoint, map_location=device, weights_only=True
        )
        teacher_state = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )
        teacher.load_state_dict(teacher_state)
        teacher.eval()

    sem_criterion = build_loss(args.loss)
    bnd_criterion = build_loss(args.bnd_loss)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    best_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        running_kd = 0.0
        train_progress = tqdm(
            train_loader,
            desc=f"Train {epoch + 1}/{args.epochs}",
            leave=False,
        )
        for batch_index, (img, mask) in enumerate(train_progress, start=1):
            img, mask = img.to(device), mask.to(device)
            bnd = make_boundary(mask)

            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                out = model(img, gts=[mask, mask])
                supervised_loss = (
                    sem_criterion(out["pred_semantic"], mask)
                    + args.bnd_w * bnd_criterion(out["pred_boundary"], bnd)
                    + args.aux_w * out["supervision_loss"]
                )
                kd_loss = torch.zeros((), device=device)
                if teacher is not None:
                    with torch.no_grad():
                        teacher_out = teacher(img)
                    kd_loss = distillation_loss(
                        out["pred_semantic"],
                        teacher_out["pred_semantic"],
                        args.kd_temperature,
                    ) + distillation_loss(
                        out["pred_boundary"],
                        teacher_out["pred_boundary"],
                        args.kd_temperature,
                    )
                loss = supervised_loss + args.kd_w * kd_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            running_kd += kd_loss.item()
            postfix = {"loss": f"{running / batch_index:.4f}"}
            if teacher is not None:
                postfix["kd"] = f"{running_kd / batch_index:.4f}"
            train_progress.set_postfix(postfix)

        sched.step()
        metrics = evaluate(
            model,
            val_loader,
            device,
            progress_desc=f"Validation {epoch + 1}/{args.epochs}",
        )
        log = (
            f"epoch {epoch + 1}/{args.epochs} loss {running / len(train_loader):.4f}"
        )
        if teacher is not None:
            log += f" KD {running_kd / len(train_loader):.4f}"
        log += (
            f" | "
            f"val IoU {metrics['iou']:.4f} F1 {metrics['f1']:.4f} "
            f"P {metrics['precision']:.4f} R {metrics['recall']:.4f}"
        )
        tqdm.write(log)

        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch,
                "model_type": "teacher" if args.mode == "standard" else "student",
                "mode": args.mode,
            },
            f"{args.out}/last.pth",
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), f"{args.out}/best.pth")
            print(f"  saved best.pth (val F1 {best_f1:.4f})")


if __name__ == "__main__":
    main()
