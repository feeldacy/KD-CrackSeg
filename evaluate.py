import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from data.registry import get_pairs
from dataset import SegDataset
from modules.metrics import EvalMetrics
from teacher_lactnet import LACTNet


def save_visualization(image_path, target, prediction, output_path):
    """Save the resized input alongside ground-truth and predicted overlays."""
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
    canvas = Image.new("RGB", (panel_width * len(panels), panel_height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = index * panel_width
        draw.text((x + 4, 4), label, fill="black")
        canvas.paste(panel, (x, 24))
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained LACTNet checkpoint on a dataset's test split."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name from datasets.yaml")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to best.pth")
    parser.add_argument("--data_root", default=None, help="Override DATA_ROOT for relative dataset roots")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save metrics as JSON")
    parser.add_argument(
        "--vis_dir",
        type=Path,
        default=None,
        help="Optional directory for a single input/ground-truth/prediction visualization",
    )
    parser.add_argument(
        "--vis_index",
        type=int,
        default=0,
        help="Zero-based test sample index to visualize (default: 0)",
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
    if args.vis_index < 0 or args.vis_index >= len(test_pairs):
        parser.error(f"--vis_index must be between 0 and {len(test_pairs) - 1}")
    test_set = SegDataset(test_pairs, mask_threshold, img_size=256, train=False)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = LACTNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    metrics = EvalMetrics(thr=args.threshold)
    if args.vis_dir is not None:
        args.vis_dir.mkdir(parents=True, exist_ok=True)
    sample_index = 0
    with torch.inference_mode():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)["pred_semantic"]
            metrics.update(logits, masks)
            batch_end = sample_index + images.shape[0]
            if args.vis_dir is not None and sample_index <= args.vis_index < batch_end:
                batch_index = args.vis_index - sample_index
                prediction = (
                    (torch.sigmoid(logits[batch_index]) > args.threshold)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
                target = (masks[batch_index] > 0.5).squeeze(0).cpu().numpy()
                image_path = test_pairs[args.vis_index][0]
                output_name = f"{args.vis_index:04d}_{image_path.stem}.png"
                save_visualization(
                    image_path,
                    target,
                    prediction,
                    args.vis_dir / output_name,
                )
            sample_index = batch_end

    results = metrics.compute()
    print(f"Dataset: {args.dataset} (test)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Images: {len(test_set)}")
    print(f"Threshold: {args.threshold:.2f}")
    print("Metrics:")
    for name, value in results.items():
        print(f"  {name}: {value:.6f}")
    if args.vis_dir is not None:
        print(f"Saved visualizations to {args.vis_dir}")

    if args.output is not None:
        report = {
            "dataset": args.dataset,
            "split": "test",
            "checkpoint": str(args.checkpoint),
            "images": len(test_set),
            "threshold": args.threshold,
            "metrics": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
