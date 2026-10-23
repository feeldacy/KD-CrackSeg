# KD-CrackSeg

This project trains and evaluates crack segmentation models with a teacher/student distillation pipeline. The main entry points are:

- `train.py` for training
- `evaluate.py` for testing and validation metrics

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
```

If a dataset root is relative, pass it with `--data_root` or set the environment variable:

```powershell
$env:DATA_ROOT = "D:/data"
```

The project matches image and mask pairs automatically by dataset and split name.

## Training commands

Use `train.py --help` to see the full list of options.

### 1) Train the teacher model

```powershell
python train.py --dataset DeepCrack --data_root D:/data --mode standard --epochs 100 --bs 8 --lr 1e-4 --out runs/exp1
```

This trains the main teacher model (`TeacherLACTNet`).

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
--vis_index VIS_INDEX             Index of the sample to visualize (default: 0)
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

## Useful CLI checks

```powershell
python train.py --help
python evaluate.py --help
```

These commands print the full list of available options for both scripts.
