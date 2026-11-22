# KD-CrackSeg

This project trains and evaluates crack segmentation models with a teacher/student distillation pipeline. The main entry points are:

- `train.py` for training
- `evaluate.py` for testing and validation metrics

The architecture of both networks — how the Swin and Conv branches are fused, what the DUC decoder and LAM do, how distillation is applied — is documented in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Environment setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you are using a different Python environment, make sure the required packages from `requirements.txt` are installed.

## Dataset configuration

Datasets are defined in `datasets.yaml`.

Example configuration:

```yaml
DeepCrack:
  root: D:/DeepCrack/
  mask_threshold: 127
  img_dir:
    train: train_img
    test: test_img
  mask_dir:
    train: train_lab
    test: test_lab

Crack500:
  root: D:/CRACK500/
  mask_threshold: 127
  lists:
    train: train.txt
    val: val.txt
    test: test.txt

Concrete:
  root: E:/concrete-v8
  mask_threshold: 127
  mask_ext: .png
  img_dir:
    train: images/train
    val: images/valid
    test: images/test
  mask_dir:
    train: masks/train
    val: masks/val
    test: masks/test
  prepare:
    kind: coco_segmentation
    annotation: _annotations.coco.json
    label_dir: labels/{split}
    size: 256
    supersample: 4
    coverage_threshold: 25
    skip_negatives: true
```

If a dataset root is relative, pass it with `--data_root` or set the environment variable:

```powershell
$env:DATA_ROOT = "D:/data"
```

The project matches image and mask pairs automatically by dataset and split name.

### Concrete dataset (Roboflow COCO-segmentation export)

`Concrete` is a Roboflow export of concrete-surface defects with five classes
(`crack`, `hole`, `honeycombe`, `pitting`, `spalling`). It is laid out as:

```
E:/concrete-v8/
  images/
    train/    *.jpg + _annotations.coco.json
    valid/    *.jpg + _annotations.coco.json
    test/     *.jpg + _annotations.coco.json
  masks/      train/ val/ test/   <- generated
  labels/     train/ val/ test/   <- generated
```

The polygons live next to the images of each split in `_annotations.coco.json`.
They are turned into mask PNGs **automatically the first time the dataset is
used** — `data/registry.py` sees the `prepare:` block in `datasets.yaml`,
notices that the masks are missing and converts them on the spot. Converting
all three splits takes about 14 seconds, so there is nothing to run by hand.

To convert ahead of time, or to change how it is done, use the module:

```powershell
python -m data.prepare --src E:/concrete-v8
```

It writes `masks/<split>/*.png` (binary foreground, all classes merged) and
`labels/<split>/*.png` (uint8 label map, 0 = background, 1-5 = class id). The
label maps are not used by the binary pipeline, they only keep the per-class
information for a future multi-class run.

The conversion is deterministic: regenerating the masks reproduces the
existing ones pixel for pixel.

Details worth knowing:

- **Polygons are rasterised at the training resolution.** LACTNet runs at
  256x256 while the source images are 1280x1280, so the converter draws every
  polygon at 4x the target size and area-downsamples it back. Measured against
  a mask rasterised at full resolution, this loses 0.17 % of the foreground
  pixels and produces fewer disconnected fragments than a nearest-neighbour
  downsample. Change it with `size`, `supersample` and `coverage_threshold`
  in the `prepare:` block of `datasets.yaml`, or with the matching command
  line flags.
- **Images without any annotation are excluded** from the training set
  (103 train, 5 valid, 1 test). Set `skip_negatives: false` in the `prepare:`
  block, or pass `--keep_negatives`, to keep them as background-only samples.
- **Overlapping annotations** are resolved by draw order: the polygon that
  comes later in the JSON wins.
- The five classes are merged into one foreground channel, so the training run
  is a binary "defect vs. background" segmentation.

## Training commands

Use `train.py --help` to see the full list of options.

### 1) Train the teacher model

```powershell
python train.py --dataset DeepCrack --data_root D:/data --mode standard --epochs 100 --bs 8 --lr 1e-4 --out runs/exp1
```

This trains the main teacher model (`TeacherLACTNet`).

On the concrete dataset the same command applies, the root is absolute so no
`--data_root` is needed:

```powershell
python train.py --dataset Concrete --mode standard --epochs 100 --bs 8 --lr 1e-4 --out runs/concrete_teacher
```

### 2) Train the student model without a teacher

```powershell
python train.py --dataset DeepCrack --data_root D:/data --mode student --epochs 100 --bs 8 --lr 1e-4 --out runs/student_exp1
```

This trains the compact student model (`StudentLACTNet`) without distillation.

### 3) Train the student model with knowledge distillation

```powershell
python train.py --dataset DeepCrack --data_root D:/data --mode distill --teacher_checkpoint runs/exp1/best.pth --epochs 100 --bs 8 --lr 1e-4 --out runs/distill_exp1 --kd_w 1.0 --kd_temperature 2.0
```

This requires `--teacher_checkpoint` and uses a frozen teacher network for distillation.

### Training arguments

`train.py` supports the following arguments:

```text
--dataset DATASET                     Required dataset name from datasets.yaml
--data_root DATA_ROOT                 Optional override for relative dataset roots
--mode {standard,distill,student}     Training mode
--teacher_checkpoint PATH             Required for distillation; accepts best.pth or last.pth
--kd_w KD_W                           Distillation weight (default: 1.0)
--kd_temperature KD_TEMPERATURE       Temperature for distillation (default: 2.0)
--epochs EPOCHS                       Number of epochs (default: 100)
--bs BS                               Batch size (default: 8)
--lr LR                               Learning rate (default: 1e-4)
--out OUT                             Output directory for checkpoints (default: runs/exp1)
--loss {dice_bce,dice_focal}          Main segmentation loss
--bnd_loss {dice_bce,dice_focal}      Boundary loss
--bnd_w BND_W                         Boundary loss weight (default: 0.5)
--aux_w AUX_W                         Auxiliary supervision weight (default: 0.5)
--val_ratio VAL_RATIO                 Validation split ratio if dataset has no built-in val set
--seed SEED                           Random seed (default: 0)
```

### Outputs from training

Each training run writes checkpoints to the chosen output directory:

- `last.pth` - latest model weights and optimizer state
- `best.pth` - best validation F1 checkpoint
- `val_files.txt` - validation file list for the run

The training script also performs validation each epoch and reports:

- loss
- validation IoU
- F1 score
- precision
- recall

## Validation commands

Validation is performed automatically during training using a held-out validation set or the dataset's built-in validation split.

### Auto validation during training

If `datasets.yaml` contains a val split, the project uses it directly. Otherwise, it creates a validation split from the training data using `--val_ratio`.

Example:

```powershell
python train.py --dataset Crack500 --data_root D:/data --mode standard --val_ratio 0.1
```

The script prints per-epoch validation metrics such as:

```text
epoch 1/100 loss 0.8231 | val IoU 0.4213 F1 0.5852 P 0.6731 R 0.5214
```

## Testing commands

Use `evaluate.py` to run evaluation on a checkpoint against the dataset's test split.

```powershell
python evaluate.py --dataset DeepCrack --checkpoint runs/exp1/best.pth --model teacher
```

### Example evaluation with custom batch size and JSON report

```powershell
python evaluate.py --dataset DeepCrack --checkpoint runs/exp1/best.pth --model teacher --batch_size 8 --threshold 0.5 --output results/metrics.json
```

### Evaluation arguments

`evaluate.py` supports the following arguments:

```text
--dataset DATASET                 Required dataset name from datasets.yaml
--checkpoint CHECKPOINT           Required path to a model checkpoint
--model {teacher,student}         Architecture type used by the checkpoint
--data_root DATA_ROOT             Override DATA_ROOT for relative dataset roots
--batch_size BATCH_SIZE           Batch size for test inference (default: 8)
--num_workers NUM_WORKERS         Number of dataloader workers (default: 4)
--threshold THRESHOLD             Binary threshold for segmentation predictions (default: 0.5)
--output OUTPUT                   Optional JSON file for saved evaluation metrics
--vis_dir VIS_DIR                 Optional visualization output directory
--vis_index VIS_INDEX [VIS_INDEX ...]
                                 Zero-based test sample indices to visualize (default: 0).
                                 Accepts several indices, e.g. --vis_index 0 5 12
--vis_count VIS_COUNT             Visualize the first N test samples. Overrides --vis_index
--vis_all                         Visualize every test sample. Overrides --vis_index
--vis_sheet                       Also write a single contact sheet PNG holding every visualization
--vis_sheet_columns VIS_SHEET_COLUMNS
                                 Number of columns in the contact sheet (default: 4)
```

### Visualizing several samples

`--vis_index` takes any number of indices, and each visualization is written
as its own PNG (`<index>_<file name>.png`) holding the input, the ground truth
in green and the prediction in red. With `--vis_sheet` the same samples are
additionally tiled into one `contact_sheet.png`:

```powershell
# the first 8 test samples, tiled into one contact sheet
python evaluate.py --dataset Concrete --checkpoint runs/teacher_exp1_concrete/last.pth --model teacher --vis_dir output/concrete_teacher/visualization --vis_count 8 --vis_sheet

# three specific samples
python evaluate.py --dataset Concrete --checkpoint runs/teacher_exp1_concrete/last.pth --model teacher --vis_dir output/concrete_teacher/vis_indices --vis_index 3 10 25

# every test sample
python evaluate.py --dataset Concrete --checkpoint runs/teacher_exp1_concrete/last.pth --model teacher --vis_dir output/concrete_teacher/all --vis_all --vis_sheet
```

### Visualization during evaluation

You can save a single sample visualization showing input, ground truth, and prediction overlay:

```powershell
python evaluate.py --dataset DeepCrack --checkpoint runs/exp1/best.pth --model teacher --vis_dir runs/vis --vis_index 0
```

The visualization output includes:

- input image
- ground truth mask overlay (green)
- prediction mask overlay (red)

## Quick-start examples

### Teacher training

```powershell
python train.py --dataset DeepCrack --data_root D:/data --mode standard --epochs 100 --bs 8 --out runs/exp1
```

### Student training

```powershell
python train.py --dataset DeepCrack --data_root D:/data --mode student --epochs 100 --bs 8 --out runs/student_exp1
```

### Distillation training

```powershell
python train.py --dataset DeepCrack --data_root D:/data --mode distill --teacher_checkpoint runs/exp1/best.pth --epochs 100 --bs 8 --out runs/distill_exp1
```

### Evaluate a trained checkpoint

```powershell
python evaluate.py --dataset DeepCrack --checkpoint runs/exp1/best.pth --model teacher --output results/deepcrack_metrics.json
```

## Model complexity

`model_complexity.py` reports parameter count, estimated FLOPs and inference
FPS for the LACTNet teacher and student.

```powershell
python model_complexity.py --model all --device cuda
```

## Useful CLI checks

```powershell
python train.py --help
python evaluate.py --help
```

These commands print the full list of available options for both scripts.
