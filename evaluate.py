import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from data.registry import get_pairs
from dataset import SegDataset
from modules.metrics import EvalMetrics
from student_lactnet import LACTNet as StudentLACTNet
from teacher_lactnet import LACTNet as TeacherLACTNet


def save_visualization(image_path, target, prediction, output_path, title=None):
    """Save the resized input alongside ground-truth and predicted overlays.

    Returns the canvas so it can be reused for a contact sheet.
    """
    size = target.shape
    image = Image.open(image_path).convert("RGB").resize(
        (size[1], size[0]), Image.Resampling.BILINEAR
    )
    base = np.asarray(image).copy()
    target_mask = target.astype(bool)
    prediction_mask = prediction.astype(bool)

    target_overlay = base.copy()
    target_overlay[target_mask] = (0, 255, 0)
    prediction_overlay = base.copy()
    prediction_overlay[prediction_mask] = (255, 0, 0)

    panels = [
        image,
        Image.fromarray((base * 0.5 + target_overlay * 0.5).astype(np.uint8)),
        Image.fromarray((base * 0.5 + prediction_overlay * 0.5).astype(np.uint8)),
    ]
    panel_width, panel_height = image.size
    labels = ("Input", "Ground truth (green)", "Prediction (red)")
    header = 44 if title else 24
    canvas = Image.new("RGB", (panel_width * len(panels), panel_height + header), "white")
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((4, 4), title, fill="black")
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = index * panel_width
        draw.text((x + 4, header - 20), label, fill="black")
        canvas.paste(panel, (x, header))
    canvas.save(output_path)
    return canvas


def save_contact_sheet(canvases, output_path, columns=4):
    """Tile several visualizations into a single image.

    Every canvas already carries its own title strip, so the sheet keeps the
    file name of each sample visible.
    """
    if not canvases:
        return
    width, height = canvases[0].size
    rows = math.ceil(len(canvases) / columns)
    sheet = Image.new("RGB", (width * columns, height * rows), "white")
    for index, canvas in enumerate(canvases):
        sheet.paste(canvas, ((index % columns) * width, (index // columns) * height))
    sheet.save(output_path)
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained LACTNet checkpoint on a dataset's test split."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name from datasets.yaml")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to best.pth")
    parser.add_argument(
        "--model",
        choices=["teacher", "student"],
        default="teacher",
        help="Architecture used by the checkpoint",
    )
    parser.add_argument("--data_root", default=None, help="Override DATA_ROOT for relative dataset roots")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save metrics as JSON")
    parser.add_argument(
        "--vis_dir",
        type=Path,
        default=None,
        help="Optional directory for the input/ground-truth/prediction visualizations",
    )
    parser.add_argument(
        "--vis_index",
        type=int,
        nargs="+",
        default=[0],
        help="Zero-based test sample indices to visualize (default: 0). "
             "Accepts several indices, e.g. --vis_index 0 5 12",
    )
    parser.add_argument(
        "--vis_count",
        type=int,
        default=None,
        help="Visualize the first N test samples. Overrides --vis_index",
    )
    parser.add_argument(
        "--vis_all",
        action="store_true",
        help="Visualize every test sample. Overrides --vis_index",
    )
    parser.add_argument(
        "--vis_sheet",
        action="store_true",
        help="Also write a single contact sheet PNG holding every visualization",
    )
    parser.add_argument(
        "--vis_sheet_columns",
        type=int,
        default=4,
        help="Number of columns in the contact sheet (default: 4)",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num_workers cannot be negative")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_pairs, mask_threshold = get_pairs(args.dataset, "test", data_root=args.data_root)
    total_samples = len(test_pairs)

    # Decide which test samples get visualized.
    if args.vis_count is not None:
        if args.vis_count < 1:
            parser.error("--vis_count must be at least 1")
        selected = list(range(min(args.vis_count, total_samples)))
    elif args.vis_all:
        selected = list(range(total_samples))
    else:
        selected = list(args.vis_index)
    for index in selected:
        if index < 0 or index >= total_samples:
            parser.error(f"--vis_index must be between 0 and {total_samples - 1}")
    selected = set(selected)

    if args.vis_sheet_columns < 1:
        parser.error("--vis_sheet_columns must be at least 1")
    test_set = SegDataset(test_pairs, mask_threshold, img_size=256, train=False)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model_cls = TeacherLACTNet if args.model == "teacher" else StudentLACTNet
    model = model_cls().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    metrics = EvalMetrics(thr=args.threshold)
    if args.vis_dir is not None:
        args.vis_dir.mkdir(parents=True, exist_ok=True)
    sample_index = 0
    canvases = []
    saved_names = []
    with torch.inference_mode():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)["pred_semantic"]
            metrics.update(logits, masks)
            batch_end = sample_index + images.shape[0]
            if args.vis_dir is not None:
                for local_index in range(images.shape[0]):
                    global_index = sample_index + local_index
                    if global_index not in selected:
                        continue
                    prediction = (
                        (torch.sigmoid(logits[local_index]) > args.threshold)
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )
                    target = (masks[local_index] > 0.5).squeeze(0).cpu().numpy()
                    image_path = test_pairs[global_index][0]
                    output_name = f"{global_index:04d}_{image_path.stem}.png"
                    canvas = save_visualization(
                        image_path,
                        target,
                        prediction,
                        args.vis_dir / output_name,
                        title=output_name[:-4],
                    )
                    canvases.append(canvas)
                    saved_names.append(output_name)
            sample_index = batch_end

    if args.vis_dir is not None and args.vis_sheet and canvases:
        sheet_path = args.vis_dir / "contact_sheet.png"
        save_contact_sheet(canvases, sheet_path, columns=args.vis_sheet_columns)
        print(f"Saved contact sheet to {sheet_path}")

    results = metrics.compute()
    print(f"Dataset: {args.dataset} (test)")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Images: {len(test_set)}")
    print(f"Threshold: {args.threshold:.2f}")
    print("Metrics:")
    for name, value in results.items():
        print(f"  {name}: {value:.6f}")
    if args.vis_dir is not None:
        print(f"Saved {len(saved_names)} visualizations to {args.vis_dir}")

    if args.output is not None:
        report = {
            "dataset": args.dataset,
            "split": "test",
            "model": args.model,
            "checkpoint": str(args.checkpoint),
            "images": len(test_set),
            "threshold": args.threshold,
            "metrics": results,
        }
        if saved_names:
            report["visualizations"] = saved_names
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
