import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.registry import get_pairs
from dataset import SegDataset
from modules.metrics import EvalMetrics
from teacher_lactnet import LACTNet


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
    with torch.inference_mode():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)["pred_semantic"]
            metrics.update(logits, masks)

    results = metrics.compute()
    print(f"Dataset: {args.dataset} (test)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Images: {len(test_set)}")
    print(f"Threshold: {args.threshold:.2f}")
    print("Metrics:")
    for name, value in results.items():
        print(f"  {name}: {value:.6f}")

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
