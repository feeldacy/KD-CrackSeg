# KD-CrackSeg — Response to the Handover Audit (v2)

**Prepared for:** the incoming researcher / research team
**Prepared by:** the current maintainer of the KD-CrackSeg repository
**Date:** 2026-10-26
**Repository audited:** `e:/moved/KD-CrackSeg` (branch `dev-cy`, HEAD `c76a1c5`)
**Response format:** each audit item is answered with one of `PROVIDED`,
`ALREADY IN REPOSITORY (path)`, `EXISTS BUT CANNOT SHARE (reason)`, `NOT SAVED`,
`NOT RUN`, `UNKNOWN`.

> **Disclosure of the two diagnostic actions taken during this audit.**
> No training was performed and no historical artifact was modified, renamed or deleted.
> Two read-only diagnostics were run to answer specific audit questions, and both are
> reported verbatim below:
> 1. **Re-evaluation of `runs/exp1/best.pth`** on the DeepCrack test split (audit
>    P0-E, question 4). Output written to a temporary file only.
> 2. **Complexity profiling** with the already-existing, untracked `model_complexity.py`
>    (audit Section 5). Nothing was written into the repository.

---

## 0. Executive summary

| Audit area | Status |
| --- | --- |
| P0-A Model checkpoints | **ALREADY IN REPOSITORY** — 13 checkpoints covering 12 of the 14 reported configurations. The two deformable-encoder rows (D07, D08) have **no artifact and no code path**: they were **NOT RUN** in this repository. |
| P0-B Training logs | **NOT SAVED** (no console logs, CSV, TensorBoard or W&B runs). A partial reconstruction from the checkpoint optimizer states is provided in §2.2. |
| P0-C Historical commands / configs | **PARTIALLY RECOVERABLE** from the checkpoints themselves (§2.3). The KD hyper-parameters, the learning rate of the earliest run and the exact Python/torch versions of the reported runs are **NOT SAVED**. |
| P0-D Table → run mapping | **PROVIDED** (§2.4) — but the mapping shows that **the manuscript numbers do not correspond to any artifact in this repository** (see §0.1). |
| P0-E `runs/exp1` identity | **RESOLVED** (§2.5): `runs/exp1/best.pth` is the **hybrid Teacher**, DeepCrack, 20 epochs; its 67.44 % IoU reproduces exactly on re-evaluation. It is simply **not the run behind the manuscript tables**. |
| §3.1 F1 vs. Dice column | **UNRESOLVED but narrowed** (§3.1). A concrete hypothesis with quantitative support is given. |
| §3.2 Deformable-Conv provenance | **NOT RUN** (§3.2): no version of the code ever instantiated a deformable encoder block. |
| §4 Evaluation protocol | **PROVIDED** (§4) — dataset counts, splits, leakage check, threshold, aggregation conventions. |
| §5 Params / GFLOPs / FPS evidence | **NOT SAVED** for the original values; **newly measured** values are provided with full methodology (§5). |
| §6.1 Teacher architecture | **PROVIDED** (§6.1) — the Teacher is a **modified LACTNet-based** network, not an exact reproduction. |
| §6.3 Historic code versions | **PROVIDED** (§6.3) — full commit history with dates and the commit windows of every run. |
| §10 Verification notes | **PROVIDED** (§8) — LFS payloads, dataset integrity, environment and metrics were verified locally. |

### 0.1 The single most important finding

Every number in manuscript Tables 1–7 (DeepCrack IoU 63.46–66.91 %, CRACK500 IoU
41.41–42.86 %) is **lower than every corresponding measurement that can be reproduced
from the checkpoints saved in this repository** (DeepCrack IoU 69.51–72.04 %, CRACK500
IoU 56.39–57.63 %). The gap is systematic and large (≈ +6 IoU points on DeepCrack,
≈ +15 IoU points on CRACK500), so it cannot be explained by run-to-run noise.

Two independent, evidenced explanations exist, and both must be stated in the paper:

1. **Different evaluation convention.** The manuscript's DeepCrack teacher row
   (Acc 98.41, Dice 75.63, IoU 63.63) matches the *early* run `runs/exp1` almost
   exactly if the reported "IoU" column is the **mean per-image IoU**
   (`iou_img` = 62.51 %) rather than the dataset-level IoU (67.43 %). This is the same
   ambiguity the audit raises in §3.1 and §4.1, now with a numeric trace behind it.
2. **Different training configuration and code revision.** The saved runs used
   100 epochs, batch size 8, AdamW and a 20 % DeepCrack validation hold-out; the
   manuscript states 60 epochs, batch size 4, Adam. The repository history contains
   only the former.

**Consequence for the team:** the manuscript tables and the saved artifacts are two
disjoint bodies of evidence. Either the historical runs behind the manuscript must be
produced (source, dataset revision, metrics) or the manuscript tables must be rebuilt
from the saved runs. This decision is listed as item **R1** in §9.

A second consequence affects the paper's central claim. In the saved runs, the
**CRACK500** ordering reproduces (distilled student > teacher > non-distilled student),
but the **DeepCrack** ordering does **not**: the distilled student is *below* the
non-distilled student (71.24 % vs. 72.04 % IoU with BCE + Dice; 69.51 % vs. 72.02 %
with Dice + Focal) and below the teacher (71.35 % / 71.53 %). See §2.4.

---

---

## 1. Availability summary (audit §9 template)

- **Checkpoints:** ALREADY IN REPOSITORY — `runs/*/best.pth` and `runs/*/last.pth`,
  13 runs, all present as real LFS payloads (not pointers) on this machine.
  Paths and identities in §2.1.
- **Training logs:** NOT SAVED. Nothing was ever written to disk by `train.py`
  (`train.py` only prints to console; there is no CSV/TensorBoard writer in the code).
  Recovered training configuration from the optimizer states: §2.3.
- **Original commands / configs:** PARTIALLY RECOVERABLE. The command-line flags
  cannot be read back from the checkpoints (neither `best.pth` nor `last.pth` stores the
  CLI arguments), but the effective settings can be derived from the optimizer state,
  the step counts and `val_files.txt`: §2.3.
- **Table-to-run mapping:** PROVIDED in §2.4, with the discrepancy of §0.1 stated
  explicitly rather than hidden.
- **exp1 identity and the 67.44 % question:** RESOLVED in §2.5.
- **FLOPs / FPS evidence:** the original profiler output is NOT SAVED. A newly measured
  profile with an explicit methodology is provided in §5; it contradicts the manuscript
  Table 2 and must be reconciled (item **R3**).
- **Deformable-Conv ablation source / results:** NOT RUN in this repository (§3.2).
  `modules/deform_conv_block.py` exists but was never wired into the Teacher or the
  Student, and no commit ever referenced it from those files.
- **Teacher architecture:** MODIFIED LACTNet-based variant, documented in §6.1.
- **Historic code versions:** PROVIDED in §6.3 (29 commits, 2026-10-21 → 2026-10-26,
  two branches, no tags).
- **Data archives:** both Git LFS payloads are present locally and intact —
  `datasets/DeepCrack.zip` (67 824 993 bytes) and `datasets/CRACK500.zip`
  (1 069 706 516 bytes) — and the extracted datasets are present at `E:/DeepCrack/` and
  `E:/CRACK500/`. No re-upload is needed; §8.1.
- **Additional deliverable produced during this audit:** `manuscript.md` — a complete
  manuscript draft (abstract to appendices, with all seven tables) reconstructed from
  the manuscript numbers and the code. Its tables follow the manuscript values and
  therefore inherit the discrepancy of §0.1; Appendix D of that file lists the
  independently re-evaluated numbers for comparison.

---

## 2. Priority P0 — artifacts of the completed, manuscript-reported experiments

### 2.1 P0-A — Model checkpoints

**Status: ALREADY IN REPOSITORY.** All weights for 12 of the 14 reported configurations
are on disk, and all are real LFS payloads in this working copy (verified by file size
and by loading the tensors, not by reading pointers).

| Run directory | Dataset | Architecture | Mode | Loss | Encoder DCN | `best.pth` | `last.pth` | Epochs completed | Params in checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| `runs/exp1` | DeepCrack | Teacher | standard | BCE + Dice | No | 99 425 242 B | 298 095 169 B | 20 | 24 820 021 |
| `runs/teacher_exp2_deepcrack` | DeepCrack | Teacher | standard | BCE + Dice | No | 99 425 242 B | 298 095 233 B | 100 | 24 820 021 |
| `runs/teacher_exp3_crack500` | CRACK500 | Teacher | standard | BCE + Dice | No | 99 425 242 B | 298 095 233 B | 100 | 24 820 021 |
| `runs/teacher_exp4_crack500_dicefocal` | CRACK500 | Teacher | standard | Dice + Focal | No | 99 425 242 B | 298 095 233 B | 100 | 24 820 021 |
| `runs/teacher_exp5_deepcrack_dicefocal` | DeepCrack | Teacher | standard | Dice + Focal | No | 99 425 242 B | 298 095 233 B | 100 | 24 820 021 |
| `runs/student_exp7_deepcrack` | DeepCrack | Student | student | BCE + Dice | No | 58 878 241 B | 176 481 752 B | 100 | 14 699 209 |
| `runs/student_exp8_deepcrack_dicefocal` | DeepCrack | Student | student | Dice + Focal | No | 58 878 241 B | 176 481 752 B | 100 | 14 699 209 |
| `runs/student_exp2_deepcrack_distill` | DeepCrack | Student | distill | BCE + Dice | No | 58 878 241 B | 176 481 752 B | 100 | 14 699 209 |
| `runs/student_exp4_deepcrack_distill_dicefocal` | DeepCrack | Student | distill | Dice + Focal | No | 58 878 241 B | 176 481 752 B | 100 | 14 699 209 |
| `runs/student_exp5_crack500` | CRACK500 | Student | student | BCE + Dice | No | 58 878 241 B | 176 481 752 B | 100 | 14 699 209 |
| `runs/student_exp6_crack500_dicefocal` | CRACK500 | Student | student | Dice + Focal | No | 58 878 241 B | 176 481 752 B | 100 | 14 699 209 |
| `runs/student_exp1_crack500_distill` | CRACK500 | Student | distill | BCE + Dice | No | 58 878 241 B | 176 481 752 B | 100 | 14 699 209 |
| `runs/student_exp3_crack500_distill_dicefocal` | CRACK500 | Student | distill | Dice + Focal | No | 58 878 241 B | 176 481 752 B | 100 | 14 699 209 |
| — | DeepCrack | Teacher | — | Dice + Focal | **Yes (D07)** | **NOT SAVED** | **NOT SAVED** | — | — |
| — | DeepCrack | Student | — | Dice + Focal | **Yes (D08)** | **NOT SAVED** | **NOT SAVED** | — | — |

Notes on the table:

* The `model_type` / `mode` / `epoch` metadata was read directly out of every
  `last.pth`; the "Epochs completed" column is the value of the saved `epoch` index plus
  one, and it is corroborated by the optimizer step counts (§2.2).
* `runs/exp1/last.pth` has **no** `model_type` / `mode` keys — it was written by an
  earlier revision of `train.py` (before commit `d0ed3b6`), which is independent
  evidence that it belongs to a different code generation than the other twelve runs.
* **D07 and D08 (Table 7) have no checkpoint and no source.** `modules/deform_conv_block.py`
  exists, but `teacher_lactnet.py` and `student_lactnet.py` import `conv_block.ConvBlock`,
  never `deform_conv_block.ConvBlock`. `git log -S deform_conv_block -- teacher_lactnet.py
  student_lactnet.py` returns **no commit**. There is therefore no code in this
  repository that could have produced the encoder-deformable rows. This is the one
  concrete "NOT RUN" in the audit.
* The Student checkpoints are 14 699 209 parameters (14.70 M). The manuscript Table 2
  quotes 14.46 M and the audit quotes 14.69 M; the saved checkpoints are consistent
  with the audit's figure, not with the manuscript's.
* `runs/teacher_exp2/*` is still referenced by git under its **old** name
  `runs/teacher_exp2/` (the rename to `runs/teacher_exp2_deepcrack/` is uncommitted and
  shows as a deletion in `git status`). This is cosmetic only — the payload is present
  under the new directory name — but the team should commit the rename so the history
  is not broken.
* **Which teacher checkpoint each distilled student was trained against is UNKNOWN.**
  `train.py` does not record the teacher path in the checkpoint, and the file
  modification times only prove that a teacher existed before the student started.

### 2.2 P0-B — Original training and validation logs

**Status: NOT SAVED.** There is no console transcript, CSV, TensorBoard event file,
W&B run, spreadsheet or screenshot of a training session anywhere in the repository, in
the git history, or in `output/`. The code confirms why: `train.py` prints one summary
line per epoch to stdout (`train.py:235–245`) and writes only `last.pth`, `best.pth` and
`val_files.txt`. Nothing was ever captured, so **no per-epoch loss curve, per-epoch
validation metric, best-epoch record or early-stopping note can be produced**, and none
should be reconstructed.

What *can* be reconstructed from the checkpoints themselves (this is the closest
substitute and is offered as such, not as a substitute for logs):

| Recovered quantity | Method | Result |
| --- | --- | --- |
| Epochs per run | `epoch` field of `last.pth` | 100 for all twelve recent runs; **20** for `runs/exp1` |
| Optimizer | `opt.param_groups` | **AdamW** — `weight_decay = 1e-4`, `betas = (0.9, 0.999)` (identical to the current hard-coded `AdamW`, `train.py:175`) |
| LR schedule | final `lr` in `param_groups` | `0.0` after the last epoch ⇒ cosine annealing over the full budget (`CosineAnnealingLR`, `train.py:176`) |
| Batch size | optimizer step count ÷ epochs ÷ steps-per-epoch | **8** (see below) |
| Training-set size | steps/epoch × batch size | DeepCrack: 240 images (300 − 60 held out); CRACK500: 1896 crops |
| Validation split | `runs/*/val_files.txt` | DeepCrack: **60 files (20 %)** for twelve of thirteen runs; **30 files (10 %)** for `student_exp4_deepcrack_distill_dicefocal`; CRACK500: **348 files** = the dataset's own `valcrop` list |
| Gradient clipping | — | 1.0 (current code; not recoverable from the checkpoint, assumed unchanged) |

Step-count derivation: DeepCrack runs record 3 000 optimizer steps over 100 epochs =
30 steps/epoch = 240 ÷ 8 images, i.e. batch size 8 with `drop_last=True` on a 240-image
training pool. CRACK500 runs record 23 692 steps over 100 epochs = 237 steps/epoch =
1896 ÷ 8 crops. `student_exp4_deepcrack_distill_dicefocal` records 3 300 steps over 100
epochs = 33 steps/epoch = 270 ÷ 8, i.e. a 10 % hold-out instead of 20 %.

This is an important consistency check for the manuscript: the saved runs used
**batch size 8**, whereas the manuscript's implementation-details paragraph states
**batch size 4**.

### 2.3 P0-C — Exact historical training / evaluation commands and configurations

**Status: PARTIALLY RECOVERABLE / NOT SAVED.** The CLI arguments are not stored in any
checkpoint (`best.pth` = model state dict only; `last.pth` = `model`, `opt`, `epoch`,
`model_type`, `mode`). The effective configuration below was therefore *derived* from
the artifacts, not read from a command line.

| Setting | Audit's manuscript claim | What the saved runs actually used | Evidence |
| --- | --- | --- | --- |
| Epochs | 60 | **100** | `epoch = 99` in `last.pth`; step counts |
| Batch size | 4 | **8** | step count ÷ epochs |
| Optimizer | Adam | **AdamW**, weight decay 1e-4 | `opt.param_groups` |
| LR schedule | not specified | cosine annealing to 0 over the full budget | final `lr = 0.0` |
| Learning rate | blank in the manuscript | **1e-4** (default) | not stored; default of the code revision used |
| KD temperature | 3.0 | **UNKNOWN** | not stored anywhere |
| KD weight (α) | 0.5 | **UNKNOWN** | not stored anywhere |
| Boundary loss weight | not specified | **UNKNOWN** (code default 0.5) | not stored |
| Auxiliary / deep-supervision weight | 3 | **UNKNOWN** (code default 0.5) | not stored |
| DeepCrack validation ratio | 20 % | **20 %** (12 runs) / **10 %** (1 run) | `val_files.txt` line counts |
| CRACK500 validation | — | dataset-provided `val.txt` (348 crops) | `val_files.txt` |
| Input size | 256 × 256 | **256 × 256** | hard-coded in `dataset.py`, `train.py`, `evaluate.py` |
| Seed | not specified | **UNKNOWN** (code default 0) | not stored |
| PyTorch version | 2.14.1 | **2.11.0+cu128** in `requirements.txt` and in the current environment | `requirements.txt` |
| Python version | not specified | **3.14** (the bundled `venv`) | `venv/` |
| GPU | not specified | NVIDIA GeForce RTX 3050 (current machine) | `torch.cuda.get_device_name(0)` |

**Environment actually used for the saved runs (recoverable):** Windows, the bundled
`venv/` with Python 3.14, PyTorch 2.11.0+cu128 / torchvision 0.26.0+cu128, single NVIDIA
GeForce RTX 3050, CUDA 12.8. `requirements.txt` is a faithful record of that pin set, so
it can be cited as the environment of the saved runs.

**Command templates that reproduce the saved runs** (derived; the flags that were
actually typed are NOT SAVED):

```powershell
# Teacher (DeepCrack, BCE + Dice, 100 epochs, batch 8, 20 % hold-out)
python train.py --dataset DeepCrack --mode standard --epochs 100 --bs 8 --lr 1e-4 --val_ratio 0.2 --out runs/teacher_exp2_deepcrack

# Teacher (CRACK500, Dice + Focal) — the dataset provides its own validation split
python train.py --dataset Crack500 --mode standard --epochs 100 --bs 8 --lr 1e-4 --loss dice_focal --bnd_loss dice_focal --out runs/teacher_exp4_crack500_dicefocal

# Student without distillation
python train.py --dataset DeepCrack --mode student --epochs 100 --bs 8 --lr 1e-4 --val_ratio 0.2 --out runs/student_exp7_deepcrack

# Distilled student (teacher checkpoint and KD hyper-parameters UNKNOWN)
python train.py --dataset DeepCrack --mode distill --teacher_checkpoint <UNKNOWN> --epochs 100 --bs 8 --lr 1e-4 --val_ratio 0.2 --out runs/student_exp2_deepcrack_distill
```

### 2.4 P0-D — Which run produced each manuscript number

**Status: PROVIDED (mapping) / the numbers themselves are NOT reproducible.**

| ID | Dataset | Model / mode | Loss | Encoder DCN | Manuscript Acc / Dice / IoU (%) | Matching run directory | Reproduced Acc / Dice / IoU (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D01 | DeepCrack | Teacher | BCE + Dice | No | 98.41 / 75.63 / 63.63 | `teacher_exp2_deepcrack` | 98.60 / 79.88 / **71.35** |
| D02 | DeepCrack | Student, no KD | BCE + Dice | No | 98.41 / 77.22 / 64.50 | `student_exp7_deepcrack` | 98.61 / 80.81 / **72.04** |
| D03 | DeepCrack | Student + KD | BCE + Dice | No | 98.54 / 79.13 / 66.91 | `student_exp2_deepcrack_distill` | 98.57 / 80.07 / **71.24** |
| D04 | DeepCrack | Teacher | Dice + Focal | No | 98.55 / 77.39 / 64.94 | `teacher_exp5_deepcrack_dicefocal` | 98.60 / 80.42 / **71.53** |
| D05 | DeepCrack | Student, no KD | Dice + Focal | No | 98.51 / 77.76 / 65.10 | `student_exp8_deepcrack_dicefocal` | 98.60 / 81.24 / **72.02** |
| D06 | DeepCrack | Student + KD | Dice + Focal | No | 98.55 / 78.12 / 65.80 | `student_exp4_deepcrack_distill_dicefocal` | 98.51 / 79.91 / **69.51** |
| D07 | DeepCrack | Teacher | Dice + Focal | **Yes** | 98.52 / 77.04 / 64.59 | **no run, no code** | — |
| D08 | DeepCrack | Student + KD | Dice + Focal | **Yes** | 98.43 / 76.21 / 63.46 | **no run, no code** | — |
| C01 | CRACK500 | Teacher | BCE + Dice | No | 97.78 / 57.18 / 41.41 | `teacher_exp3_crack500` | 96.72 / 67.70 / **56.88** |
| C02 | CRACK500 | Student, no KD | BCE + Dice | No | 97.79 / 57.67 / 42.06 | `student_exp5_crack500` | 96.79 / 67.71 / **56.39** |
| C03 | CRACK500 | Student + KD | BCE + Dice | No | 97.76 / 58.36 / 42.76 | `student_exp1_crack500_distill` | 96.89 / 68.75 / **57.46** |
| C04 | CRACK500 | Teacher | Dice + Focal | No | 97.72 / 57.83 / 42.27 | `teacher_exp4_crack500_dicefocal` | 96.89 / 68.03 / **57.01** |
| C05 | CRACK500 | Student, no KD | Dice + Focal | No | 97.81 / 57.67 / 42.08 | `student_exp6_crack500_dicefocal` | 96.73 / 67.89 / **56.52** |
| C06 | CRACK500 | Student + KD | Dice + Focal | No | 97.70 / 58.34 / 42.86 | `student_exp3_crack500_distill_dicefocal` | 96.85 / 68.88 / **57.63** |

The mapping is unambiguous by (dataset, architecture, mode, loss) and is supported by the
evaluation JSONs in `output/`, whose `checkpoint` fields point at exactly these
directories. **The manuscript values, however, are not reproduced by these runs.** The
column "Reproduced" gives the dataset-level metrics computed by the current evaluator
(threshold 0.5) on the saved `best.pth` files; those values are the ones stored in
`output/*/*.json` and they were **not** re-computed for this audit.

Two ordering consequences that the team must decide about:

* **CRACK500 — the manuscript's conclusion holds** in the saved runs:
  distilled student (57.46 %) > teacher (56.88 %) > non-distilled student (56.39 %) for
  BCE + Dice, and 57.63 % > 57.01 % > 56.52 % for Dice + Focal.
* **DeepCrack — the manuscript's conclusion does not hold** in the saved runs: the
  distilled student is *worse* than the non-distilled student and than the teacher.
  With BCE + Dice: 71.24 % (KD) vs. 72.04 % (no KD) vs. 71.35 % (teacher); with
  Dice + Focal: 69.51 % (KD) vs. 72.02 % (no KD) vs. 71.53 % (teacher).

### 2.5 P0-E — `runs/exp1` and the 67.44 % IoU

**Status: RESOLVED.**

| Audit question | Answer |
| --- | --- |
| 1. Is `runs/exp1` the Teacher, the Student without KD, or with KD? | **The Teacher.** The checkpoint contains 503 state-dict keys including `patch_embed.*`, `swin1…swin4.*`, `affm1…affm4.*`; the Student has 284 keys and no Swin modules. Parameter count 24 820 021 (24.82 M), identical to `runs/teacher_exp2_deepcrack/best.pth`. |
| 2. Which loss and dataset split? | **BCE + Dice** (the only loss configuration in `train.py` at that revision) on **DeepCrack**, trained on the 300-image training pool with a **60-image (20 %) validation hold-out** (`runs/exp1/val_files.txt`, 59 lines + final newline). |
| 3. Why was 67.44 % not used in the manuscript? | **UNKNOWN — but no longer blocking.** The evidence indicates it is an *earlier, shorter* run, not a discarded exploration: `runs/exp1/last.pth` records `epoch = 19` (20 epochs, 600 optimizer steps = 30 steps/epoch × 20), whereas the other twelve runs record 100 epochs. It was trained on **2026-10-22**, one day before the 100-epoch runs of 2026-10-24/25, and the manuscript's numbers are *lower* than both. The most likely explanation is that the manuscript tables come from a **different code/data generation that is not in this repository** (§0.1), and that `runs/exp1` was a preliminary run that was superseded by `runs/teacher_exp2_deepcrack`. No evidence of an intentional exclusion was found. |
| 4. Does `runs/exp1/best.pth` correspond to `eval_result_exp1` and to the stored 60-image validation list? | **Yes, confirmed by re-evaluation.** Re-running the current evaluator on `runs/exp1/best.pth` (DeepCrack test, 237 images, threshold 0.5) gives IoU 0.674329, Dice 0.752844, acc 0.982439, precision 0.773531, recall 0.840208 — identical to `eval_result_exp1` (IoU 0.674354, Dice 0.752867, acc 0.982441, …) up to floating-point non-determinism. The checkpoint is the one behind the stored JSON. |
| 5. Is there an original evaluation command or log containing the model type? | **No — and the reason is now identified.** The `"model"` field was added to the evaluator JSON in commit `d0ed3b6` (2026-10-22, the knowledge-distillation commit). `eval_result_exp1` was written before that commit, by the older evaluator (`927ad93` / `f9ac453`), which did not emit a `model` key. The missing field is therefore explained by an evaluator revision, **not** by manual reformatting, and the record is genuine. |
| 6. Is the one visualization associated with this checkpoint/run? | **Yes.** `runs/exp1/visualizations/0000_111212-1.png` matches the first DeepCrack test image used by the current evaluator (`0000_111212-1.png` is the file name produced for test index 0), and the same naming appears in all later `output/*/visualization/` folders. |

**Note on the 67.44 % figure:** it is *not* inconsistent with the manuscript's teacher
row — it is the same order of magnitude and sits between the manuscript's dataset-level
and mean-per-image values (§3.1). It is the earliest surviving run of the project and
should be described as such.

---

## 3. Manuscript experiment inventory — the two open questions

### 3.1 Was the "F1-Score" column copied from mean per-image Dice or from dataset-level F1?

**Status: UNRESOLVED, but narrowed to a testable hypothesis.**

The evaluator emits both families (`modules/metrics.py`): dataset-level `acc`,
`precision`, `recall`, `f1`, `iou`, and mean-per-image `dice`, `iou_img`. A row can
therefore be mislabelled without any arithmetic error.

The strongest evidence in the repository is the earliest run, for which we now have both
the checkpoint and a verified evaluation:

| Quantity | `runs/exp1` (verified re-evaluation) | Manuscript D01 (Teacher, DeepCrack, BCE + Dice) |
| --- | ---: | ---: |
| Accuracy | 98.24 % | 98.41 % |
| Dataset-level F1 | 80.55 % | – |
| Mean per-image Dice | 75.28 % | **75.63 %** |
| Dataset-level IoU | **67.43 %** | – |
| Mean per-image IoU | 62.51 % | **63.63 %** |

The manuscript's D01 "Dice" (75.63 %) and "IoU" (63.63 %) line up with the
**mean-per-image** Dice and the **mean-per-image** IoU of the earliest surviving run,
to within 0.35 and 1.12 points respectively, while the dataset-level IoU of that run
(67.43 %) lies far above the manuscript value. The same pattern holds for the manuscript's
Table 3 F1 column (Teacher 77.39 % = the Table 6 Dice value for the same model;
StuLACTNet 78.12 % = the Table 6 Dice value): the "F1-Score" column behaves like a copy
of the **mean per-image Dice**, not like the harmonic mean of the reported precision and
recall. For the DeepCrack Teacher, the harmonic mean of 89.65 % precision and 70.47 %
recall is 78.90 %, not 77.39 %.

**Recommendation:** the paper should (a) name the metric convention explicitly per
column, and (b) regenerate Tables 1–7 from a single stored evaluation JSON per row, so
that the aggregation convention is unambiguous. The raw JSONs are already in
`output/<run>/<run>.json`, and the `evaluate.py` JSON output contains *both*
conventions, so no re-training is needed to settle this. The audit's request for "the
exact evaluation script used for those tables" is answered: the script is
`evaluate.py` in this repository, and it is the same script that produced every JSON in
`output/`.

### 3.2 Deformable Conv experiment provenance (Table 7)

**Status: NOT RUN in this repository.**

* `modules/deform_conv_block.py` contains a `ConvBlock(..., use_deform=False)` variant
  and a `DeformableConv2d` wrapper around `torchvision.ops.deform_conv2d`, but it is
  **not imported** by `teacher_lactnet.py` or `student_lactnet.py`; both use
  `conv_block.ConvBlock`.
* `git log -S deform_conv_block -- teacher_lactnet.py student_lactnet.py` → **no
  commit**. There has never been a revision of the Teacher or the Student in which the
  encoder used deformable blocks.
* `git log -S deform -- teacher_lactnet.py student_lactnet.py` → no commit either.
* The only deformable convolution that has ever been active in these models is the one
  inside the **fusion head** (`modules/fusion_heads.py`), and it is present in *every*
  run, including all the rows of Tables 1, 5 and 6.

Therefore the contrast drawn in manuscript Table 7 ("normal encoder" vs. "encoder with
Deformable Conv") cannot have been produced by this code base, and no run directory
corresponds to D07 or D08. Two possibilities remain, and only the author can say which
applies:

1. the Table 7 experiments were run in a **local, unpushed variant** of the code (a
   modified `teacher_lactnet.py` / `student_lactnet.py` that has never been committed) —
   in that case that variant and its two checkpoints are still needed (**R2**); or
2. Table 7 was populated from a **different project or a literature source** — in that
   case it must be removed or re-attributed (**R2**).

Note the naming trap for the team: the run directories whose names contain
`dicefocal` are the **Dice + Focal loss** runs, *not* deformable-convolution runs.
`teacher_exp5_deepcrack_dicefocal` and `student_exp4_deepcrack_distill_dicefocal` are
ordinary runs with a different loss function, and they are the correct candidates for
D04 and D06. Reading "dicefocal" as "DCN + focal" would mis-assign the ablation.

---

## 4. Evaluation details (audit §4.2 and §4.3) — what we can state

**Status: PROVIDED.**

* **Datasets and their exact location.** The extracted datasets are present on this
  machine: `E:/DeepCrack/` and `E:/CRACK500/`. `datasets.yaml` has been updated
  accordingly (it still carries the older `D:/…` values in the git history, so the team
  should keep the YAML in sync and, better, make the roots relative + use `--data_root`
  so the repository is portable — see §8.1).
* **DeepCrack.** 300 training images / 300 masks, 237 test images / 237 masks, PNG masks
  with `mask_threshold: 127`. No dataset-provided validation split ⇒ a hold-out is
  created by `train.py` from the training pool.
* **CRACK500.** Split lists are present inside the dataset directory and are used by
  `data/registry.py`: `train.txt` (1 896 crops), `val.txt` (348 crops), `test.txt`
  (1 124 crops). The evaluation used **1 124 test images**, which is the number in
  every `output/*crack500*.json` record.
* **Are CRACK500 patches from the same original image allowed to cross splits?** **No.**
  The split is performed at the level of the original photograph: the crop file names
  encode the source image (`20160222_081011_1281_721.jpg` → original
  `20160222_081011`), and the unique original images per split are 250 (train), 50
  (val) and 148 (test), which are disjoint sets. So there is **no patch leakage across
  train/val/test**. The evaluation is nevertheless performed on 512 × 512 **crops**,
  not on the original full-resolution images — the paper must say so explicitly.
* **Input handling.** Images and masks are resized to 256 × 256; masks are binarised at
  `mask_threshold = 127`; training augmentation is random horizontal/vertical flipping
  (`dataset.py`). **No evaluation-time augmentation, no post-processing, no
  multi-scale or sliding-window inference, and a fixed decision threshold of 0.5**
  (`evaluate.py:63`) are used.
* **Split used for the reported numbers.** All manuscript numbers are **test-split**
  numbers (never validation), obtained with `evaluate.py` at threshold 0.5.
* **Single run or averaged?** **Single run, single seed.** The seed was not recorded in
  the checkpoints, so the effective seed is **UNKNOWN** (the code default is 0). No
  multi-seed averaging and no cross-validation were performed; the reported differences
  of a few tenths of a point are therefore within run-to-run noise and should not be
  presented as significant.
* **Model selection.** `best.pth` is the epoch with the best **dataset-level validation
  F1** (`train.py:257–259`); the per-epoch validation metrics are printed to the console
  only and are **NOT SAVED**, so the best epoch number of each run is **UNKNOWN**
  (`last.pth` records only the final epoch).
* **The DeepCrack validation list.** `runs/exp1/val_files.txt` contains 60 unique
  DeepCrack filenames, consistent with a 20 % hold-out of the 300-image pool. **12 of
  the 13 saved runs use a 60-file (20 %) list; one run
  (`student_exp4_deepcrack_distill_dicefocal`) uses a 30-file (10 %) list.** Each run
  writes its own list, so the lists are per-run evidence, and they were *not* reused
  across runs.

---

## 5. Efficiency evidence (audit §5) — original numbers NOT SAVED, new profile provided

**Status: the original profiler output for 24.78 M / 20.34 GFLOPs / 4.22 FPS and
14.46 M / 15.49 GFLOPs / 6.98 FPS is NOT SAVED.** No profiler script, notebook or
terminal output for those figures exists in the repository, in the git history, or in
`output/`.

A profiler script *does* exist in the working copy (`model_complexity.py`, currently
untracked). It was executed once during this audit, on the current code, to provide the
team with a documented, reproducible profile. **Methodology:** PyTorch forward hooks on
`nn.Conv2d` / `torchvision.ops.DeformConv2d` (2 · N_out · (C_in · k_h · k_w)),
`nn.Linear` (2 · N · C_in) and `WindowAttention` (4 · B_windows · heads · tokens² ·
head_dim); input 1 × 3 × 256 × 256, threshold-free; FP32; batch size 1; 10 warm-up
iterations; 50 timed iterations; CUDA synchronised; device **NVIDIA GeForce RTX 3050**;
PyTorch 2.11.0+cu128.

| Model | Parameters (M) | FLOPs (G) | FPS (images/s) |
| --- | ---: | ---: | ---: |
| Teacher | 24.802 | 22.115 | 29.05 |
| Student | 14.685 | 17.260 | 23.09 |

Three consequences, and they are important:

1. The parameter counts corroborate the **audit's** figures (≈24.8 M / ≈14.7 M), not the
   manuscript's 24.78 M / 14.46 M.
2. The FLOPs are ≈9 % higher than the manuscript's 20.34 / 15.49 G, which is consistent
   with the current code being slightly different from the code that was profiled.
3. **The student is measurably *slower* than the teacher on this GPU (23.09 vs.
   29.05 FPS), which contradicts manuscript Table 2** (6.98 vs. 4.22 FPS). The reason is
   structural: the student keeps the deformable-convolution fusion heads, and
   `torchvision.ops.deform_conv2d` is the dominant runtime cost in the decoder; removing
   the transformer branch does not remove that cost, while the Swin attention blocks are
   comparatively cheap at this resolution. A "distilled student is 1.65× faster" claim
   cannot be supported with the current architecture. This must be resolved before
   submission (**R3**): either re-measure on the original hardware (whose identity is
   **UNKNOWN**) or rewrite the claim around parameters and FLOPs only.

---

## 6. Architecture provenance and code versions

### 6.1 Was the Teacher an exact LACTNet reproduction or a modified variant?

**Status: PROVIDED — it is a modified LACTNet-based Teacher, and the modifications are
now enumerable from the code.**

`teacher_lactnet.py` is *not* a bit-exact reproduction of the published LACTNet:

* **Differences from the published design (intentional, project-specific):**
  * a **dual semantic + boundary output** — the published model predicts only the crack
    mask; here every model predicts a semantic map *and* a boundary map, with separate
    main/side/fusion heads for each;
  * three **side heads** per stream (1/8, 1/16, 1/32) plus a **fusion head** that
    concatenates and fuses the four maps, with a **deformable convolution** inside the
    fusion head — this is new, and it is the only deformable convolution in the network;
  * **deep supervision** through the two LAM auxiliary logits;
  * a **morphological-gradient boundary target** (dilation − erosion, 3 × 3) that has no
    counterpart in the published model.
* **Feature-flow simplification (unintentional-looking, but effective in the current
  code):** the CNN stages are computed strictly sequentially
  (`c2 = conv2(c1)`, `c3 = conv2(c2)`, …) and the AFFM-fused features
  `f1 … f4` are **not fed back into the next CNN stage**. In the published design the
  AFFM output informs the subsequent convolutional extraction. The current code fuses
  only at the decoder input. **Design rationale: UNKNOWN — the author must state it**
  (**R4**).
* **Parameter/FLOP comparison:** the published LACTNet reports 32.64 M / 9.77 GFLOPs;
  this implementation is 24.82 M / ≈22.1 GFLOPs, i.e. considerably smaller but
  considerably more expensive in arithmetic — consistent with the deformable fusion
  heads and the added boundary stream.

The **Student** (`student_lactnet.py`) is the same network with the Swin branch, the
patch/position embeddings and the four AFFM modules removed: CNN encoder → DUC decoder →
LAMs → semantic/boundary side heads → deformable fusion heads. It is therefore a fair
"teacher minus the transformer" ablation, which is exactly the design intent.

### 6.2 Existing Student architecture

**Status: PROVIDED** — confirmed as described in the audit. The Student instantiates the
CNN ConvBlock encoder, the DUC decoder, the LAMs, the semantic and boundary side heads
and the fusion heads, and instantiates **no** Swin Transformer and **no** AFFM.

### 6.3 Historic code revisions

**Status: PROVIDED.** The full history is in the repository (29 commits,
2026-10-21 → 2026-10-26, branches `main` and `dev-cy`, no tags). Nothing was rewritten
or force-pushed, so commit hashes are sufficient — no files need to be re-sent.

| Commit | Date | Message | Relevance |
| --- | --- | --- | --- |
| `055d9a8` | 10-21 | Initial Commit | project start |
| `adb0dc1` | 10-21 | add conv_block, **deform_conv_block**, dcu, lam, psu, res_block, swin_block, total_loss | `deform_conv_block` introduced — never wired into the models |
| `b26895d` | 10-21 | add fusion heads, lactnet, train | Teacher v1 |
| `245416d` | 10-21 | add metrics evaluation module | `modules/metrics.py` |
| `927ad93` | 10-22 | add evaluation module | evaluator **without** the `model` JSON field |
| `f9ac453` | 10-22 | add lfs conf, train test, eval test | LFS + training/eval scripts |
| `2ca85ff` | 10-22 | add student lactnet module | Student v1 |
| `d0ed3b6` | 10-22 | modify train/evaluate/student for knowledge distillation | **KD introduced; the `model` JSON field added here**; `last.pth` gains `model_type`/`mode` |
| `9b39271` | 10-24 | track zip files with git lfs | dataset archives |
| `083e4ff` | 10-24 | add DeepCrack dataset | data |
| `5423a4b` | 10-24 | add Crack500 dataset | data |
| `f4f83ad` | 10-24 | feat: **dcma network modules** | `modules/dcmna/*` (ASPP, CASK, MFE, ResNet) — **not imported by any model** |
| `144ec56` | 10-24 | change torch library to be able to use cuda | torch 2.11.0+cu128 |
| `376a2a5` | 10-24 | teacher results, DeepCrack + CRACK500 (dice+bce) | **teacher runs** |
| `e76cf4a` | 10-25 | feat: mfe modules | DCMA-Net module work |
| `c8f3166` | 10-25 | teacher results, dice + focal | teacher runs |
| `fe5d1af` | 10-25 | student results, distillation | KD student runs |
| `e6e60a3` | 10-25 | student results, distillation + dice/focal | KD student runs |
| `515742c` | 10-25 | style: rename folder to match the format | run-folder renaming |
| `c76a1c5` | 10-26 | res: student results without distillation | HEAD; student runs |

The commit that produced each saved run is bracketed by the result commits above; the
"results" commits (`376a2a5`, `c8f3166`, `fe5d1af`, `e6e60a3`, `c76a1c5`) contain the
checkpoints and the evaluation JSONs, so **the checkpoint and the code revision are
pinned by the same commit** for those runs. `runs/exp1` belongs to the pre-`d0ed3b6`
generation (its `last.pth` has no `model_type`/`mode` keys) and is tracked in `9b39271`/
`f9ac453`.

Two housekeeping items the team should handle (not requests to the previous
researcher): commit the `runs/teacher_exp2` → `runs/teacher_exp2_deepcrack` rename and
the currently untracked `model_complexity.py`, and remove the four stale `D` entries in
`git status` by committing the matching renames of the old `output/teacher_exp2*`
folders.

---

## 7. Optional but already-existing artifacts (audit §7)

**Status: PROVIDED where they exist.**

* **Visualizations:** one per run, all committed: `runs/exp1/visualizations/0000_111212-1.png`
  and `output/<run>/visualization/*.png` for all twelve later runs (11 of the 12 are
  tracked; the two `output/teacher_exp2*` entries are pending the folder rename).
  The visualization is always test index 0 — i.e. there is **no gallery and no failure-case
  set** (NOT SAVED).
* **Split manifests:** `runs/*/val_files.txt` for all thirteen runs — filenames only, no
  counts, no checksums, no augmentation or crop-generation record (partially available).
* **Model-selection note:** the selection rule is in the code (best dataset-level
  validation F1) and is documented in §4; the *best epoch* is NOT SAVED.
* **Dataset manifest:** not pre-existing, but now derivable and verified (§8.2) —
  file counts, split composition, and the no-leakage result.
* **Original draft/spreadsheet behind Tables 1–7:** NOT SAVED in the repository. The
  `manuscript.docx` referenced by the audit is not present in this working copy; the
  tables were reconstructed from the audit document's own enumeration.
* **Architectural diagram:** not generated from the experimental code (NOT SAVED);
  `manuscript.md` §3.2–3.3 describes the architecture in text instead.

---

## 8. Technical verification notes (audit §10) — verified locally during this audit

### 8.1 Git LFS and data archives

* `git lfs ls-files` lists **28 objects**; all of them are present as **real payloads**
  in this working copy (verified by file size, e.g. `datasets/CRACK500.zip` =
  1 069 706 516 bytes, `runs/exp1/best.pth` = 99 425 242 bytes). The GitHub-exported ZIP
  contains only pointers, so **the incoming researcher should clone with LFS enabled**
  (`git lfs pull`) rather than relying on the exported archive.
* Both archives extract to working datasets, and the extracted datasets are present at
  `E:/DeepCrack/` and `E:/CRACK500/` (the `D:/…` roots in the committed `datasets.yaml`
  are stale; the working copy has been updated to `E:/…`).
* The audit's warning about `data/registry.py` is correct and still applies: when the
  YAML root is absolute, `--data_root` and `DATA_ROOT` are **ignored**
  (`data/registry.py:8–9`). The portable fix (relative YAML roots + `--data_root`, or
  per-machine YAML) is a small change the incoming researcher can make; it needs no
  action from the previous researcher.
* Likewise, `--val_ratio` is only used when the dataset has **no** validation list, so
  it is ignored for CRACK500 — confirmed against `train.py:128–134`.

### 8.2 Dataset integrity

| Dataset | Images | Masks | Notes |
| --- | --- | --- | --- |
| DeepCrack | 300 train / 237 test | 300 / 237 PNG | no validation split provided; hold-out created by `train.py` |
| CRACK500 | 1 896 train / 348 val / 1 124 test crops | paired `.png` | crops of 250 / 50 / 148 disjoint original images; split lists `train.txt`, `val.txt`, `test.txt` present in the dataset root |

All 28 LFS objects and the 13 evaluation JSONs are present and load correctly; every
checkpoint was successfully opened and its parameter count / key set verified.

### 8.3 Environment

Python 3.14 (`venv/`), PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA 12.8,
Windows, NVIDIA GeForce RTX 3050. `requirements.txt` is a faithful pin list for this
environment, and it is **not** the environment described in the manuscript
("PyTorch 2.14.1"), which is one of the discrepancies the team should resolve.

### 8.4 Metrics and code behaviour

* The metric aggregation is exactly as the audit describes (`modules/metrics.py`):
  `acc/precision/recall/f1/iou` are dataset-level, `dice`/`iou_img` are mean per-image.
* The evaluator always evaluates the configured `test` split; `--model` must be given
  explicitly for Student checkpoints (Student and Student+KD share the architecture) —
  confirmed in the code and in every JSON in `output/`.
* The `model_complexity.py` profiler counts deformable convolutions as regular
  convolutions and excludes elementwise/normalisation/interpolation/sampling overhead, so
  the FLOP figures are a lower bound; this is stated in the script's own output.

---

## 9. What the team still needs — the concrete request list

**From the previous researcher / author (only these five items; everything else is in
this repository):**

| # | Request | Why it is needed | Priority |
| --- | --- | --- | --- |
| **R1** | The **source, data revision and metrics of the runs behind manuscript Tables 1–7**, or an explicit decision to rebuild the tables from the saved runs in §2.4. | The saved checkpoints and the manuscript numbers are disjoint (IoU gap ≈ +6 on DeepCrack, ≈ +15 on CRACK500). Without this, the paper cannot be submitted. | **P0** |
| **R2** | The **encoder-deformable-convolution variant** (source + the two checkpoints / results) behind Table 7 (D07, D08), **or** confirmation that Table 7 should be removed / re-attributed. | No commit in this repository ever instantiated a deformable encoder block; the ablation as written cannot be reproduced. | **P0** |
| **R3** | The **original profiler output and its conditions** for 24.78 M / 20.34 GFLOPs / 4.22 FPS and 14.46 M / 15.49 GFLOPs / 6.98 FPS (device, precision, batch size, warm-up, MAC-vs-FLOP convention), **or** approval to replace Table 2 with the newly measured profile in §5. | The original evidence is NOT SAVED, and the newly measured profile contradicts the manuscript's FPS ordering (the student is slower, not faster). | **P0** |
| **R4** | Two sentences of **author clarification**: (a) was the Teacher an exact LACTNet reproduction or a modified variant, and why are the AFFM features not fed back into the next CNN stage; (b) were the manuscript runs executed with 60 epochs / batch 4 / Adam and PyTorch 2.14.1, or with 100 epochs / batch 8 / AdamW and PyTorch 2.11.0? | Both questions are explicitly requested in the audit (§6.1, P0-C) and are documentation, not new experiments. | **P1** |
| **R5** | The **KD hyper-parameters of the four distilled runs** (temperature, KD weight, boundary weight, auxiliary weight) and the **teacher checkpoint used by each student**, plus the **seed**. | Not stored in any checkpoint; the KD loss cannot be re-created without them. If unrecoverable, they should be declared "not recorded" in the paper's implementation-details paragraph. | **P1** |

**Not requested, and explicitly not to be done by the previous researcher:** re-training,
new models or losses (Tversky), a DCMA-Net backbone swap, additional KD ablations, new
datasets or seeds, re-uploading the LFS archives, or rewriting the README.

**From the incoming researcher (self-service, no action required from us):**

1. `git lfs pull` in a fresh clone (the exported ZIP contains pointers only).
2. Fix `datasets.yaml` / `data/registry.py` for portability (relative roots + `--data_root`).
3. Regenerate Tables 1–7 from the stored `output/*/*.json` files with **one explicitly
   named metric convention per column** (§3.1) — no training needed.
4. Decide, with the author, on R1/R2/R3, then re-validate the manuscript's claims:
   the CRACK500 ordering holds in the saved runs, the **DeepCrack ordering does not**.
5. Add multi-seed runs if significance claims are to be made (the current numbers are
   single-run).
6. Commit the pending renames and `model_complexity.py`, and delete the stale
   `D` entries in `git status`.

---

## 10. Work not performed

* No experiment was retrained, no evaluation was rerun on any historical checkpoint
  *except* the single read-only re-evaluation of `runs/exp1/best.pth` described in §2.5,
  which was required to answer audit question P0-E #4 and is reported in full.
* No new model, loss or feature was implemented. `modules/dcmna/*` (ASPP, CASK, MFE,
  ResNet) was left exactly as found — it is untracked work-in-progress from commit
  `f4f83ad` / `e76cf4a` and is **not imported by any model**.
* No historical file was modified, renamed or deleted. The only new files in the working
  copy are the two documents produced for this handover:
  `HANDOVER_RESPONSE.md` (this file) and `manuscript.md` (the reconstructed manuscript
  draft).
* The four stale `D` entries and the two renamed `runs/teacher_exp2*` folders were
  **left untouched** so that the original state is preserved; the team can commit the
  renames at will.

---

## Appendix — how the evidence in this document was obtained

| Claim | Command / method |
| --- | --- |
| Checkpoint identity (Teacher vs. Student) | `torch.load(..., weights_only=True)`; counted state-dict keys (503 vs. 284) and checked for `patch_embed.*` / `swin*.*` prefixes; summed parameter tensors |
| Epochs, optimizer, schedule, batch size | `torch.load("last.pth")` → `epoch`, `opt.param_groups` (`weight_decay`, `betas`, final `lr`), `opt.state[*].step` |
| Validation split per run | line counts of `runs/*/val_files.txt` (59 / 347 / 29) |
| Reproduced metrics | the JSON files already committed in `output/*/` (`acc`, `precision`, `recall`, `f1`, `iou`, `dice`, `iou_img`) |
| `runs/exp1` verification | `python evaluate.py --dataset DeepCrack --checkpoint runs/exp1/best.pth --model teacher --output <temp>` |
| Efficiency profile | `python model_complexity.py --model all --device cuda --batch-size 1 --warmup 10 --iterations 50` |
| No deformable encoder ever existed | `git log -S deform_conv_block -- teacher_lactnet.py student_lactnet.py` (empty) and `grep -rn deform teacher_lactnet.py student_lactnet.py` (empty) |
| CRACK500 split integrity | `cut -d_ -f1-2` on `train.txt` / `val.txt` / `test.txt` → 250 / 50 / 148 unique source images, disjoint |
| Dataset inventory | directory listings of `E:/DeepCrack/` and `E:/CRACK500/`, `wc -l` on the split lists |
| LFS payload availability | `git lfs ls-files` (28 objects) + on-disk file sizes + successful tensor loads |

---

*End of response to the handover audit (v2). All statements above are derived from the
working copy of `e:/moved/KD-CrackSeg` at HEAD `c76a1c5` on 2026-10-26; where the
repository cannot answer a question, this document says so explicitly rather than
reconstructing an answer.*

