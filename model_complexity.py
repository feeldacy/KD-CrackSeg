import argparse
import time

import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

from modules.swin_transformer import WindowAttention
from student_lactnet import LACTNet as StudentLACTNet
from teacher_lactnet import LACTNet as TeacherLACTNet


MODEL_CLASSES = {
    "teacher": TeacherLACTNet,
    "student": StudentLACTNet,
}
IMAGE_SIZE = 256
MODEL_INPUT_SIZES = {
    "teacher": (IMAGE_SIZE, IMAGE_SIZE),
    "student": (IMAGE_SIZE, IMAGE_SIZE),
}


def _multiply_add_flops(module, output):
    output_elements = output.numel()
    weights_per_output = module.weight.numel() // module.weight.shape[0]
    return 2 * output_elements * weights_per_output


def _attention_matmul_flops(module, inputs, _output):
    batch_windows, tokens, channels = inputs[0].shape
    head_dim = channels // module.num_heads
    return 4 * batch_windows * module.num_heads * tokens**2 * head_dim


def count_flops(model, sample):
    """Count conv/linear and attention matmul FLOPs with forward hooks."""
    total_flops = 0
    handles = []

    def count_conv(_module, _inputs, output):
        nonlocal total_flops
        total_flops += _multiply_add_flops(_module, output)

    def count_linear(_module, _inputs, output):
        nonlocal total_flops
        total_flops += 2 * output.numel() * _module.in_features

    def count_attention(_module, inputs, output):
        nonlocal total_flops
        total_flops += _attention_matmul_flops(_module, inputs, output)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, DeformConv2d)):
            handles.append(module.register_forward_hook(count_conv))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(count_linear))
        elif isinstance(module, WindowAttention):
            handles.append(module.register_forward_hook(count_attention))

    try:
        with torch.inference_mode():
            model(sample)
    finally:
        for handle in handles:
            handle.remove()

    return total_flops


def measure_fps(model, sample, device, warmup, iterations):
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(iterations):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    return sample.shape[0] * iterations / elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Print parameter count, estimated FLOPs, and inference FPS for LACTNet models."
    )
    parser.add_argument(
        "--model",
        choices=["teacher", "student", "all"],
        default="all",
        help="Model to measure (default: all)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Images per inference batch; use 1 for per-image latency (default: 1)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup inference iterations before timing (default: 10)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Timed inference iterations (default: 50)",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    model_names = MODEL_CLASSES if args.model == "all" else [args.model]
    results = []

    for name in model_names:
        model = MODEL_CLASSES[name]().to(device).eval()
        height, width = MODEL_INPUT_SIZES[name]
        sample = torch.randn(
            args.batch_size, 3, height, width, device=device
        )
        parameters = sum(parameter.numel() for parameter in model.parameters())
        flops = count_flops(model, sample)
        fps = measure_fps(
            model, sample, device, args.warmup, args.iterations
        )
        results.append((name, parameters, flops, fps))
        del model, sample

    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print(
        f"{'Model':<10} {'Parameters (M)':>16} {'FLOPs (G)':>12} "
        f"{'FPS (images/s)':>16}"
    )
    for name, parameters, flops, fps in results:
        print(
            f"{name:<10} {parameters / 1e6:>16.3f} "
            f"{flops / 1e9:>12.3f} {fps:>16.2f}"
        )
    print(
        "FLOPs are estimated for convolution, linear, and attention matrix "
        "multiplications; deformable convolutions are counted as regular "
        "convolutions. Elementwise, normalization, interpolation, and "
        "deformable sampling overhead are excluded."
    )


if __name__ == "__main__":
    main()
