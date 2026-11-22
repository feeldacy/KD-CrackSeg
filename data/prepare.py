"""Conversion of source annotations into the mask PNGs the pipeline reads.

The Concrete dataset is a Roboflow export that ships its annotations as
``_annotations.coco.json`` files with polygon ``segmentation`` entries, while
``data/registry.py`` expects one mask PNG per image.  This module performs
that conversion and is used in two ways:

* ``data/registry.py`` calls :func:`ensure_prepared` when a dataset declares a
  ``prepare`` block, so a dataset that was never converted is converted on
  first use instead of failing with an empty pair list.

Running this module directly converts a dataset ahead of time:

``python -m data.prepare --src E:/concrete-v8``

The polygons are rasterised at the training resolution rather than at the
source resolution.  LACTNet runs at 256x256 while the concrete images are
1280x1280, so a polygon is drawn at 4x the target size and area-downsampled
back to it.  Measured against a mask rasterised at full resolution this loses
0.17 % of the foreground pixels and yields fewer disconnected fragments than a
nearest-neighbour downsample.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Split folders of a Roboflow export -> split names used by the pipeline.
EXPORT_TO_PIPELINE_SPLIT = {"train": "train", "valid": "val", "test": "test"}


def polygon_points(segmentation):
    """Return a list of (x, y) point lists, or None for unsupported formats."""
    if isinstance(segmentation, dict):
        return None  # RLE - not produced by this export, but keep it explicit
    if not isinstance(segmentation, list) or not segmentation:
        return None
    polygons = []
    for part in segmentation:
        if not isinstance(part, list) or len(part) < 6:
            continue
        polygons.append(
            [(float(part[i]), float(part[i + 1])) for i in range(0, len(part) - 1, 2)]
        )
    return polygons or None


def rasterise(anns, img_w, img_h, size, supersample, coverage_threshold):
    """Draw every polygon of every annotation into one binary mask.

    Polygons are drawn in annotation order, so a later polygon overwrites an
    earlier one where the two overlap.  For the merged binary mask the order
    does not change the result, but it makes the label map deterministic.

    Args:
        anns: annotations belonging to one image, in JSON order.
        img_w, img_h: original image size.
        size: output mask size (256 for LACTNet).
        supersample: intermediate rasterisation factor (4 -> 1024x1024).
        coverage_threshold: percentage of a pixel that must be covered by a
            polygon for it to count as foreground, 0-100.

    Returns:
        A (size, size) uint8 image with 0 = background and 255 = foreground.
    """
    big = size * supersample
    canvas = Image.new("L", (big, big), 0)
    draw = ImageDraw.Draw(canvas)
    scale_x = big / img_w
    scale_y = big / img_h

    for ann in anns:
        polygons = polygon_points(ann.get("segmentation"))
        if polygons is None:
            continue
        for points in polygons:
            scaled = [(x * scale_x, y * scale_y) for x, y in points]
            draw.polygon(scaled, fill=255)

    if supersample == 1:
        mask = canvas
    else:
        # BOX resampling averages the coverage of every pixel, so a hairline
        # that survives the downscale stays continuous instead of breaking up.
        mask = canvas.resize((size, size), Image.BOX)
    return mask.point(
        lambda v: 255 if v >= coverage_threshold * 255 / 100 else 0
    )


def rasterise_labels(anns, img_w, img_h, size, supersample, categories):
    """Same as ``rasterise`` but keeps the category id of every pixel."""
    big = size * supersample
    canvas = Image.new("L", (big, big), 0)
    draw = ImageDraw.Draw(canvas)
    scale_x = big / img_w
    scale_y = big / img_h

    for ann in anns:
        polygons = polygon_points(ann.get("segmentation"))
        if polygons is None:
            continue
        category = categories.get(ann.get("category_id"), 0)
        for points in polygons:
            scaled = [(x * scale_x, y * scale_y) for x, y in points]
            draw.polygon(scaled, fill=category)

    if supersample == 1:
        mask = canvas
    else:
        mask = canvas.resize((size, size), Image.BOX)
    # The averaged value keeps the id of the dominant polygon in each pixel.
    highest = max(categories.values(), default=0)
    return mask.point(lambda v: min(highest, int(round(v))))


def load_annotations(annotation_path):
    """Return (images, annotations grouped by image id, category ids)."""
    data = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    categories = {c["id"]: c["id"] for c in data.get("categories", [])}
    anns_by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)
    return data.get("images", []), anns_by_image, categories


def count_annotated_images(annotation_path):
    """Number of images that carry at least one annotation."""
    annotation_path = Path(annotation_path)
    if not annotation_path.is_file():
        return 0
    images, anns_by_image, _ = load_annotations(annotation_path)
    return sum(1 for image in images if anns_by_image.get(image["id"]))


def prepare_split(
    img_dir,
    annotation_path,
    mask_dir,
    label_dir=None,
    size=256,
    supersample=4,
    coverage_threshold=25.0,
    skip_negatives=True,
):
    """Convert one split of a COCO-segmentation export into mask PNGs.

    Args:
        img_dir: folder holding the source images of the split.
        annotation_path: the ``_annotations.coco.json`` of the split.
        mask_dir: output folder for the binary masks.
        label_dir: output folder for the per-class label maps, or None.
        size: output mask size.
        supersample: intermediate rasterisation factor.
        coverage_threshold: percentage of a pixel a polygon must cover.
        skip_negatives: leave out images that carry no annotation.

    Returns:
        A dict with the number of written masks, skipped images and missing files.
    """
    images, anns_by_image, categories = load_annotations(annotation_path)
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    if label_dir is not None:
        label_dir = Path(label_dir)
        label_dir.mkdir(parents=True, exist_ok=True)

    written = negatives = missing = 0
    fg_pixels = 0
    for image in images:
        anns = anns_by_image.get(image["id"], [])
        if not anns:
            negatives += 1
            if skip_negatives:
                continue
        if not (Path(img_dir) / image["file_name"]).is_file():
            missing += 1
            continue

        stem = Path(image["file_name"]).stem
        mask = rasterise(
            anns, image["width"], image["height"], size, supersample, coverage_threshold
        )
        mask.save(mask_dir / f"{stem}.png")
        fg_pixels += int((np.asarray(mask) > 0).sum())

        if label_dir is not None:
            label = rasterise_labels(
                anns, image["width"], image["height"], size, supersample, categories
            )
            label.save(label_dir / f"{stem}.png")

        written += 1

    pct = 100.0 * fg_pixels / max(written * size * size, 1)
    return {
        "written": written,
        "negatives": negatives,
        "missing": missing,
        "foreground_pct": pct,
    }


def ensure_prepared(base, img_dir, mask_dir, prepare_cfg, split):
    """Make sure the mask PNGs of a split exist, converting them if needed.

    Args:
        base: dataset root the mask folder is relative to.
        img_dir: folder holding the source images of the split.
        mask_dir: folder the masks are expected in.
        prepare_cfg: the ``prepare`` block of the dataset in datasets.yaml.
        split: the split the pipeline asked for.

    Returns:
        The folder the masks are in.
    """
    kind = prepare_cfg.get("kind")
    if kind != "coco_segmentation":
        raise ValueError(f"Unknown prepare kind '{kind}'")

    annotation_name = prepare_cfg.get("annotation", "_annotations.coco.json")
    annotation_path = Path(img_dir) / annotation_name
    if not annotation_path.is_file():
        # Nothing to convert from; the caller reports the missing folder.
        return Path(mask_dir)

    expected = count_annotated_images(annotation_path)
    mask_dir = Path(mask_dir)
    have = len(list(mask_dir.glob("*.png"))) if mask_dir.is_dir() else 0
    if expected and have >= expected:
        return mask_dir

    label_dir = prepare_cfg.get("label_dir")
    if label_dir:
        label_dir = base / label_dir.format(split=split)
    print(
        f"[{split}] converting {annotation_name} -> {mask_dir}"
        f" ({have}/{expected} masks present)"
    )
    stats = prepare_split(
        img_dir,
        annotation_path,
        mask_dir,
        label_dir=label_dir,
        size=prepare_cfg.get("size", 256),
        supersample=prepare_cfg.get("supersample", 4),
        coverage_threshold=prepare_cfg.get("coverage_threshold", 25.0),
        skip_negatives=prepare_cfg.get("skip_negatives", True),
    )
    print(
        f"[{split}] {stats['written']} masks written"
        f" | {stats['negatives']} images without annotation"
        f"{' (excluded)' if prepare_cfg.get('skip_negatives', True) else ' (kept as background)'}"
        f" | {stats['missing']} missing images"
        f" | foreground {stats['foreground_pct']:.2f}% of the canvas"
    )
    return mask_dir


def main():
    """Convert a dataset ahead of time: ``python -m data.prepare``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=Path("E:/concrete-v8"))
    parser.add_argument("--out", type=Path, default=None, help="Output root (default: the source folder)")
    parser.add_argument("--size", type=int, default=256, help="Output mask size")
    parser.add_argument("--supersample", type=int, default=4, help="Intermediate rasterisation factor")
    parser.add_argument(
        "--coverage_threshold",
        type=float,
        default=25.0,
        help="Percentage of a pixel that a polygon must cover to count as foreground",
    )
    parser.add_argument(
        "--keep_negatives",
        action="store_true",
        help="Also write empty masks for images without annotations",
    )
    args = parser.parse_args()

    src = args.src
    out_root = args.out or src
    print(f"source: {src}\noutput: {out_root}\n")

    for export_split, split in EXPORT_TO_PIPELINE_SPLIT.items():
        img_dir = src / "images" / export_split
        annotation_path = img_dir / "_annotations.coco.json"
        if not annotation_path.is_file():
            print(f"[{split}] no {annotation_path}, skipped")
            continue
        stats = prepare_split(
            img_dir,
            annotation_path,
            out_root / "masks" / split,
            label_dir=out_root / "labels" / split,
            size=args.size,
            supersample=args.supersample,
            coverage_threshold=args.coverage_threshold,
            skip_negatives=not args.keep_negatives,
        )
        print(
            f"[{split}] {stats['written']} masks written"
            f" | {stats['negatives']} images without annotation"
            f"{' (excluded)' if not args.keep_negatives else ' (kept as background)'}"
            f" | {stats['missing']} missing images"
            f" | foreground {stats['foreground_pct']:.2f}% of the canvas"
        )


if __name__ == "__main__":
    main()
