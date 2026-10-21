import os, yaml
from pathlib import Path

def get_pairs(name, split, yaml_path="datasets.yaml", data_root=None):
    cfg = yaml.safe_load(open(yaml_path))[name]
    root_cfg = Path(cfg["root"])
    if root_cfg.is_absolute():
        base = root_cfg                                  # absolute root in YAML: ignore DATA_ROOT
    else:
        data_root = data_root or os.environ.get("DATA_ROOT")
        if not data_root:
            raise RuntimeError(
                f"'{name}' has a relative root. Set DATA_ROOT or pass --data_root.")
        base = Path(data_root) / root_cfg

    pairs = []
    if "lists" in cfg:                                   # strategy A: list file
        for line in (base / cfg["lists"][split]).read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                pairs.append((base / parts[0], base / parts[1]))
    else:                                                # strategy B: scan folders
        img_dir = base / cfg["img_dir"][split]
        mask_dir = base / cfg["mask_dir"][split]
        for img_p in sorted(img_dir.iterdir()):
            mask_p = mask_dir / (img_p.stem + cfg["mask_ext"])
            if mask_p.exists():
                pairs.append((img_p, mask_p))

    if not pairs:
        raise FileNotFoundError(f"No pairs found for {name}/{split}")
    bad = [p for pr in pairs[:20] for p in pr if not p.exists()]
    if bad:
        raise FileNotFoundError(f"Paths don't resolve, e.g. {bad[0]}")
    print(f"[{name}/{split}] {len(pairs)} pairs")
    return pairs, cfg["mask_threshold"]

    