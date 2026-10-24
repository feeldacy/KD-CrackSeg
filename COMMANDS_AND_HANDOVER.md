# KD-CrackSeg Command Reference and Handover Notes

This file documents the commands that are available in the repository for training, validation, and evaluation. It is based on the current code in `train.py`, `evaluate.py`, `datasets.yaml`, and the project audit notes.

## 1. Important handover note

The repository contains historical experiment artifacts and Git LFS dataset pointers. Before creating or requesting new outputs:

- do not rerun training or evaluation unless specifically needed
- do not overwrite original experiment files or checkpoints
- preserve the existing archived outputs as evidence
- treat the documented commands below as the current repo workflow, not as proof that every command was historically executed

## 2. Environment setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Dataset configuration

The project reads dataset layouts from `datasets.yaml`.

```yaml
_defaults: &defaults
    mask_threshold: 127

DeepCrack:
  <<: *defaults
  root: D:/DeepCrack/
  mask_ext: .png
  img_dir:
    train: train_img
    test: test_img
  mask_dir:
    train: train_lab
    test: test_lab

Crack500:
  <<: *defaults
  root: D:/CRACK500/
  lists:
    train: train.txt
    val: val.txt
    test: test.txt
```

If a dataset root is relative, you can provide it with `--data_root` or by setting:

```powershell
$env:DATA_ROOT = "D:/data"
```

The code resolves image/mask pairs through `data/registry.py`.

## 4. Training commands

Use the main training entry point:

```powershell
python train.py --help
```

Full CLI options:

```text
usage: train.py [-h] --dataset DATASET [--data_root DATA_ROOT]
                [--mode {standard,distill,student}]
                [--teacher_checkpoint TEACHER_CHECKPOINT] [--kd_w KD_W]
                [--kd_temperature KD_TEMPERATURE] [--epochs EPOCHS] [--bs BS]
                [--lr LR] [--out OUT] [--loss {dice_bce,dice_focal}]
                [--bnd_loss {dice_bce,dice_focal}] [--bnd_w BND_W]
                [--aux_w AUX_W] [--val_ratio VAL_RATIO] [--seed SEED]
```

### 4.1 Train the teacher model

```powershell
python train.py --dataset DeepCrack --data_root D:/data --mode standard --epochs 100 --bs 8 --lr 1e-4 --out runs/exp1
```

This trains the teacher model (`TeacherLACTNet`).

### 4.2 Train the student model without a teacher

```powershell
python train.py --dataset DeepCrack --data_root D:/data --mode student --epochs 100 --bs 8 --lr 1e-4 --out runs/student_exp1
```

This trains the compact student model (`StudentLACTNet`) without distillation.

### 4.3 Train the student model with distillation

```powershell
python train.py --dataset DeepCrack --data_root D:/data --mode distill --teacher_checkpoint runs/exp1/best.pth --epochs 100 --bs 8 --lr 1e-4 --out runs/distill_exp1 --kd_w 1.0 --kd_temperature 2.0
```

This mode requires `--teacher_checkpoint` and uses a frozen teacher to guide the student.

### 4.4 Common training arguments

```text
--dataset DATASET                     Required dataset name from datasets.yaml
--data_root DATA_ROOT                 Optional override for relative dataset roots
--mode {standard,distill,student}     Training mode
--teacher_checkpoint PATH             Required when --mode distill
--kd_w KD_W                           Distillation weight (default: 1.0)
--kd_temperature KD_TEMPERATURE       Distillation temperature (default: 2.0)
--epochs EPOCHS                       Number of training epochs (default: 100)
--bs BS                               Batch size (default: 8)
--lr LR                               Learning rate (default: 1e-4)
--out OUT                             Output directory for checkpoints and logs
--loss {dice_bce,dice_focal}          Main semantic loss
--bnd_loss {dice_bce,dice_focal}      Boundary loss
--bnd_w BND_W                         Boundary loss weight (default: 0.5)
--aux_w AUX_W                         Auxiliary loss weight (default: 0.5)
--val_ratio VAL_RATIO                 Validation split ratio if no dataset val split exists
--seed SEED                           Random seed for reproducibility
```

### 4.5 Training outputs

The script saves checkpoints in the selected output directory:

- `last.pth` — latest weights and optimizer state
- `best.pth` — best validation F1 checkpoint
- `val_files.txt` — validation file list for that run

During training, validation metrics are printed every epoch. The script logs:

- loss
- validation IoU
- F1 score
- precision
- recall

## 5. Validation workflow

Validation is performed automatically during training.

- if the dataset has its own validation split, it is used
- otherwise, the code creates a validation split from the training set using `--val_ratio`

Example:

```powershell
python train.py --dataset Crack500 --data_root D:/data --mode standard --val_ratio 0.1
```

The script prints metrics such as:

```text
epoch 1/100 loss 0.8231 | val IoU 0.4213 F1 0.5852 P 0.6731 R 0.5214
```

## 6. Testing / evaluation commands

Use the evaluation entry point:

```powershell
python evaluate.py --help
```

Full CLI options:

```text
usage: evaluate.py [-h] --dataset DATASET --checkpoint CHECKPOINT
                   [--model {teacher,student}] [--data_root DATA_ROOT]
                   [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
                   [--threshold THRESHOLD] [--output OUTPUT]
                   [--vis_dir VIS_DIR] [--vis_index VIS_INDEX]
```

### 6.1 Evaluate a teacher checkpoint

```powershell
python evaluate.py --dataset DeepCrack --checkpoint runs/exp1/best.pth --model teacher
```

### 6.2 Evaluate a student checkpoint

```powershell
python evaluate.py --dataset DeepCrack --checkpoint runs/student_exp1/best.pth --model student
```

### 6.3 Evaluate and save metrics to JSON

```powershell
python evaluate.py --dataset DeepCrack --checkpoint runs/exp1/best.pth --model teacher --batch_size 8 --threshold 0.5 --output results/metrics.json
```

### 6.4 Generate a visualization for one sample

```powershell
python evaluate.py --dataset DeepCrack --checkpoint runs/exp1/best.pth --model teacher --vis_dir runs/vis --vis_index 0
```

This saves an image panel with:

- input image
- ground truth overlay (green)
- prediction overlay (red)

### 6.5 Evaluation arguments

```text
--dataset DATASET                 Required dataset name from datasets.yaml
--checkpoint CHECKPOINT           Required path to a model checkpoint
--model {teacher,student}         Architecture type used by the checkpoint
--data_root DATA_ROOT             Override DATA_ROOT for relative dataset roots
--batch_size BATCH_SIZE           Batch size for inference (default: 8)
--num_workers NUM_WORKERS         DataLoader worker count (default: 4)
--threshold THRESHOLD             Prediction threshold for binary segmentation (default: 0.5)
--output OUTPUT                   Optional JSON output path
--vis_dir VIS_DIR                 Optional folder for visualization output
--vis_index VIS_INDEX             Zero-based sample index for visualization
```

## 7. Quick-start examples

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

### Evaluation

```powershell
python evaluate.py --dataset DeepCrack --checkpoint runs/exp1/best.pth --model teacher --output results/deepcrack_metrics.json
```

## 8. Useful verification commands

```powershell
python train.py --help
python evaluate.py --help
```

These commands print the available training and evaluation options directly from the repository.

## 9. Audit-aware summary

The current repository includes the training and evaluation entry points, dataset registry, loss definitions, and model classes needed for the documented workflow. The README supplied in conversation is useful as a user-facing command reference, but it should be treated as documentation rather than historical evidence for which commands were actually run in prior experiments.
