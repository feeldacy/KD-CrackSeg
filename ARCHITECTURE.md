# Architecture of the LACTNet teacher/student pair

This document explains what the two networks in this repository actually are and
how they work, in enough detail to reimplement them or to defend their design
in a paper. Everything here is read off the source code; the shapes and the
parameter counts were measured, not estimated.

---

## 1. What the model does

Both networks perform **binary semantic segmentation of concrete-surface defects**
plus **boundary regression**, on 256×256 RGB tiles.

| Output | Shape | Meaning |
| --- | --- | --- |
| `pred_semantic` | (B, 1, 256, 256) | raw logits, defect (1) vs. background (0) |
| `pred_boundary` | (B, 1, 256, 256) | raw logits, defect edge vs. not |

The two heads are complementary. The semantic head answers *where is the defect*;
the boundary head answers *exactly where its edge is*. For thin, low-contrast
structures like cracks and pitting the edge is what carries the usable signal,
which is why the boundary branch is not merely an auxiliary task but a
first-class output that is fused into the semantic prediction.

Both outputs are **logits, not probabilities**. The sigmoid is applied inside
the loss wrappers (`modules/total_loss.py`) and at evaluation time, never in the
model itself.

---

## 2. The two networks at a glance

| | Teacher | Student |
| --- | --- | --- |
| File | `teacher_lactnet.py` | `student_lactnet.py` |
| Class | `LACTNet` | `LACTNet` |
| Encoder | Swin Transformer branch **+** Conv branch | Conv branch only |
| Skip connections | AFFM-fused features | raw Conv features |
| Parameters | 24.802 M | 14.685 M (−40.8 %) |
| FLOPs @256² | 22.115 G | 17.260 G (−22.0 %) |
| Throughput @256², CUDA, bs=1 | 33.65 img/s | 54.39 img/s (+61.6 %) |

The student is **not** a pruned or quantised teacher. It is a separately
defined architecture that keeps the Conv encoder and the whole decoder but
drops the transformer branch and the fusion modules. The teacher is the more
accurate, more expensive model; the student is the deployable one; knowledge
distillation is what transfers the teacher's dark-room knowledge to it.

Measured with `python model_complexity.py --model all --device cuda --batch-size 1`.

---

## 3. Data flow

### 3.1 Teacher

```
input (B,3,256,256)
│
├─ SWIN BRANCH ─────────────────────────────────┐
│  patch_embed  Conv2d(3→64, k=4, s=4) + LN      │  (B,64,64,64)
│  pos_embed    learned absolute position        │
│  swin1        W-MSA + SW-MSA                   │  (B,64,64,64)
│  swin_down1   Conv2d(64→128, k=2, s=2)         │
│  swin2        W-MSA + SW-MSA                   │  (B,32,32,128)
│  swin_down2   Conv2d(128→256, k=2, s=2)        │
│  swin3        W-MSA + SW-MSA                   │  (B,16,16,256)
│  swin_down3   Conv2d(256→512, k=2, s=2)        │
│  swin4        W-MSA + SW-MSA                   │  (B, 8, 8,512)
│                                                 │
└─ CONV BRANCH ──────────────────────────────────┤
   conv1  ConvBlock(3→32)                         │  (B, 32,128,128)
   conv2  ConvBlock(32→64)                        │  (B, 64, 64, 64)
   conv3  ConvBlock(64→128)                       │  (B,128, 32, 32)
   conv4  ConvBlock(128→256)                      │  (B,256, 16, 16)
   conv5  ConvBlock(256→512)                      │  (B,512,  8,  8)
                                                 │
   AFFM FUSION (per scale, Swin_i ⊕ Conv_{i+1}) ───┤
   affm1(s1, c2)   → (B, 64, 64, 64)              │
   affm2(s2, c3)   → (B,128, 32, 32)              │
   affm3(s3, c4)   → (B,256, 16, 16)              │
   affm4(s4, c5)   → (B,512,  8,  8)              │
                                                 │
   DECODER (DUC: pixel-shuffle upsample + skip + dilated residual)
   d1(f4, f3)  → (B,256, 16, 16)                 │
   d2(d1, f2)  → (B,128, 32, 32)                 │
   lam3(d2)    → (B,128, 32, 32) + aux logits l3  │
   d3(d2, f1)  → (B, 64, 64, 64)                 │
   lam2(d3)    → (B, 64, 64, 64) + aux logits l2  │
   d4(d3, c1)  → (B, 32,128,128)                 │
                                                 │
   HEADS
   main_sem/main_bnd   d4 → ↑2 → 1×1 conv          │  (B,1,256,256)
   side8   on d2  (1/8)                           │  (B,1, 32, 32)
   side16  on d1  (1/16)                          │  (B,1, 16, 16)
   side32  on f4  (1/32)                          │  (B,1,  8,  8)
        │
        └──► concat all four (upsampled to 256) → FusionHead (deformable conv)
                        │
                        ├─► pred_semantic (B,1,256,256)
                        └─► pred_boundary (B,1,256,256)
```

### 3.2 Student

Identical from the Conv branch down. The differences:

- no `patch_embed`, `pos_embed`, `swin1..4`, `swin_down1..3`
- no `affm1..4`; the skip features are the raw Conv outputs
  (`d1(c5,c4)`, `d2(d1,c3)`, `d3(d2,c2)`, `d4(d3,c1)`)
- `side32` reads `c5` instead of `f4` — same shape (B,512,8,8), same channel count

Everything else — DUC decoder, LAMs, heads, fusion, losses, distillation — is
identical between the two.

---

## 4. The building blocks

### 4.1 Patch embedding and position embedding

`modules/patch_embedding.py` — a strided 4×4 convolution turns the RGB image
into 64×64 tokens of 64 channels, followed by LayerNorm. It asserts the input
is divisible by 4.

`modules/position_embedding.py` — learned absolute position embeddings of
shape (1, 4096, 64), added to the token sequence.

> **The input resolution is locked to 256×256.** The position embedding table is
> sized for `img_size=256`, so feeding a different resolution either fails or
> silently truncates the table. This is also why the Concrete dataset's masks
> are rasterised at 256×256 rather than downsampled from 1280×1280.

### 4.2 Swin Transformer blocks

`modules/swin_transformer.py`. Each `SwinBlock` is two `SwinTransformerBlock`s:

1. **W-MSA** — window multi-head self attention inside non-overlapping 7×7 windows
2. **SW-MSA** — the same, but with the feature map cyclically shifted by 3 pixels
   so that information crosses window borders

Each block is pre-norm (`LayerNorm` → attention → residual, `LayerNorm` → MLP →
residual) with a 4-head MLP expansion of 4×. Window attention uses a learned
relative-position bias table.

The window size is fixed at 7 while the stage resolutions are 64, 32, 16 and 8,
so the 8×8 stage is effectively fully attended.

**Cost:** the Swin branch is 9.356 M parameters, 37.7 % of the teacher.

### 4.3 Conv blocks

`modules/conv_block.py`. Each block is:

```
5×5 Conv (stride 1) → BN → GELU
3×3 Depthwise Conv (stride 1) → 1×1 Pointwise Conv → BN → GELU
3×3 Depthwise Conv (stride 2) → 1×1 Pointwise Conv → BN → GELU
```

The 5×5 kernel gives a wide receptive field cheaply; the depthwise/pointwise
pair keeps the parameter count low; the final stride-2 depthwise convolution
does the downsampling. Every block halves the spatial resolution, so
`conv1..conv5` walk 256 → 128 → 64 → 32 → 16 → 8.

**Cost:** the Conv branch is 5.077 M parameters, 20.5 % of the teacher.

### 4.4 AFFM — asymmetric feature fusion

`modules/affm.py`. Each AFFM takes the Swin feature and the Conv feature at
the same scale and fuses them:

```
Swin  → 3×3 Depthwise Conv → 1×1 Conv → BN → ReLU  ─┐
Conv  → 1×1 Conv → BN → ReLU                       ─┴─ concat → 3×3 Depthwise → 1×1 → BN → ReLU
                                                                  │
                                                          ChannelAttention (SE) → multiply
```

The channel attention is the classic squeeze-and-excite: global average pool →
1×1 → ReLU → 1×1 → sigmoid, giving a per-channel weight. The result is
re-weighted so that channels which matter for the fused representation survive.

Each branch contributes half of the output channels (`out_channels // 2`).

Note the pairing: AFFM *i* fuses Swin stage *i* with Conv stage *i+1*, because
both sit at the same spatial resolution (Swin's stride-4 patching means Swin
stage 1 is at 64×64, which is Conv stage 2).

### 4.5 DUC — decoder unit

`modules/duc.py`, with `modules/psu_module.py` and `modules/residual_block.py`.

**PSU** (pixel-shuffle upsampling): a 3×3 convolution expands the channels by
4× and `nn.PixelShuffle` rearranges them into a 2× larger map. This is
sub-pixel convolution, which avoids the checkerboard artefacts that plain
transposed convolution produces — important when the target is a thin crack.

**ResidualBlock**: three dilated 3×3 convolutions with dilation 1, 2 and 5,
each followed by BN and ReLU, then squeeze-and-excite weighting, then a 1×1
shortcut addition and a final ReLU. The three dilation rates let one block
aggregate context at several scales without extra pooling.

A DUC unit is `PSU → concat with the skip → ResidualBlock`.

**Cost:** the four DUC units total 9.593 M parameters, 38.7 % of the teacher
and 65.3 % of the student. The decoder is the single largest cost in the
network, and it is the part the student keeps in full.

### 4.6 LAM — label-aware module

`modules/lam.py`. It is four lines of code and it does something important:

```python
logits = self.proj(U)          # 1×1 conv → (B,1,h,w) coarse mask prediction
mask   = torch.sigmoid(logits)
U_bar  = U * mask               # gate the feature map with the predicted mask
return U_bar, logits
```

The decoder feature is gated by the network's own current belief about where
the defect is. Only the channels inside a predicted defect region survive to
the next stage. This focuses the network's capacity on the target and
suppresses background clutter — and it returns the logits so the coarse mask
can be supervised directly.

The two LAM outputs `l2` (1/4 resolution, 64×64) and `l3` (1/8 resolution,
32×32) are used as **deep supervision** targets (see §6).

### 4.7 Heads and fusion

**`main_sem` / `main_bnd`** — 2× bilinear upsampling followed by a 1×1
convolution, producing full-resolution maps.

**`SideHead` ×3** (`modules/fusion_heads.py`) — one per scale, each predicting
a semantic map and a boundary map. These are the RM-1/8, RM-1/16 and RM-1/32
outputs. Deep supervision on multiple scales makes the decoder features
discriminative at every resolution, not just at the end.

**`FusionHead`** — the final combiner:

```
concat 4 maps → 3×3 Conv → BN → ReLU → DeformBlock → 1×1 Conv
```

`DeformBlock` is a modulated deformable convolution (DCNv2-style): a 3×3
convolution predicts 18 offset channels (2 offsets × 9 kernel positions) and
9 modulation masks, and the convolution samples from learned, input-dependent
positions. That lets the fusion step gather features from a shifted,
input-dependent neighbourhood, which suits elongated structures that a fixed
3×3 kernel would smear.

Both fusion heads run on the same concatenated stack — 4 channels for the
boundary maps, `4 × num_classes` for the semantic ones — and produce the two
final predictions.

---

## 5. Why two encoders

Cracks, pitting and spalling are thin, elongated, low-contrast structures
embedded in large, texture-rich backgrounds.

- The **Conv branch** has a strong local inductive bias and high spatial
  resolution early on — it resolves fine detail, but its receptive field grows
  slowly.
- The **Swin branch** provides long-range context and global structure at
  every scale — it can tell a crack from a joint between two concrete slabs,
  which is a decision that depends on context far from the crack itself.

The AFFM modules let the network decide, per scale and per channel, how much
to trust each view. The teacher is therefore not "a transformer plus a CNN" but
an explicitly adaptive fusion of a local-detail and a global-context
representation.

The student keeps only the Conv branch, which is why it is smaller but also
why distillation is needed to recover some of the teacher's context awareness.

---

## 6. How a training step works

`train.py`. One batch of `(image, mask)` goes through:

```
loss =   L_sem(pred_semantic, mask)                      # dice+bce or focal+dice
      + 0.5 · L_bnd(pred_boundary, boundary)             # dice+bce or focal+dice
      + 0.5 · L_aux(supervision_loss)                    # deep supervision on l2, l3
      +  kd_w · L_KD(student, teacher)                   # distillation only
```

- **Boundary ground truth** is derived from the mask on the fly
  (`make_boundary`): `dilate(mask) − erode(mask)` with a 3×3 max-pool, i.e.
  the one-pixel outline of the defect.
- **Deep supervision** compares the LAM logits `l2` (64×64) and `l3` (32×32)
  against the mask downsampled to their resolution with nearest-neighbour
  interpolation, using binary cross-entropy.
- **Loss weights** `--bnd_w` (0.5), `--aux_w` (0.5), `--kd_w` (1.0) are all
  command-line arguments. The available losses are `dice_bce`, `dice_focal` and
  `tversky_focal` (`modules/total_loss.py`), all binary.

---

## 7. Knowledge distillation

`train.py --mode distill` trains the student against a frozen teacher.

```python
def distillation_loss(student_logits, teacher_logits, temperature):
    student_logits = student_logits / T
    teacher_probs  = sigmoid(teacher_logits / T)
    # Bernoulli KL over [1-p, p], summed over the channel, averaged over pixels
    return KL(student || teacher) · T²
```

Both the semantic and the boundary map are distilled, so the teacher supervises
both tasks.

Three things to keep in mind:

1. **The KD loss is Bernoulli**, matching the binary formulation. It would have
   to be rewritten as a softmax KL for a multi-class model.
2. **The teacher must be trained on the same dataset.** The KD term matches
   logit distributions, so a teacher trained on DeepCrack hairlines is a poor
   supervisor for a Concrete model whose foreground is large pitting regions.
3. The `T²` factor keeps the gradient magnitudes comparable as the temperature
   changes.

---

## 8. Evaluation

`evaluate.py` computes metrics over the whole test split with
`modules/metrics.py`:

| Metric | How it is aggregated |
| --- | --- |
| `acc`, `precision`, `recall`, `f1`, `iou` | over pooled TP/FP/FN/TN of the entire split |
| `dice`, `iou_img` | mean of the per-image scores |

Note that `dice` is a **mean of per-image Dice**, which is not the same number
as `f1` computed from pooled counts. The two columns in the manuscript tables
have to be read with that in mind.

---

## 9. Parameter breakdown

Measured on the actual models:

**Teacher — 24.802 M**

| Component | Parameters | Share |
| --- | ---: | ---: |
| Swin branch (patch embed, pos embed, 4 SwinBlocks, 3 downsamplers) | 9.356 M | 37.7 % |
| Decoder (4 × DUC) | 9.593 M | 38.7 % |
| Conv branch (conv1–conv5) | 5.077 M | 20.5 % |
| AFFM ×4 | 0.761 M | 3.1 % |
| Heads (main, side, fusion) | 0.016 M | 0.1 % |

**Student — 14.685 M**

| Component | Parameters | Share |
| --- | ---: | ---: |
| Decoder (4 × DUC) | 9.593 M | 65.3 % |
| Conv branch (conv1–conv5) | 5.077 M | 34.6 % |
| Heads (main, side, fusion) | 0.016 M | 0.1 % |

The Swin branch and the AFFMs together account for 10.117 M parameters — the
entire gap between the two models.

---

## 10. File map

| File | Contains |
| --- | --- |
| `teacher_lactnet.py` | Teacher `LACTNet` |
| `student_lactnet.py` | Student `LACTNet` |
| `arch_registry.py` | architecture registry; maps a checkpoint to the right class |
| `modules/patch_embedding.py` | `PatchEmbedding` |
| `modules/position_embedding.py` | `PositionEmbedding` |
| `modules/swin_transformer.py` | `WindowAttention`, `SwinTransformerBlock`, `SwinBlock` |
| `modules/conv_block.py` | `ConvBlock` |
| `modules/affm.py` | `ChannelAttention`, `AFFM` |
| `modules/duc.py` | `DUC` |
| `modules/psu_module.py` | `PSU` (pixel-shuffle upsampling) |
| `modules/residual_block.py` | `SEBlock`, `ResidualBlock` |
| `modules/lam.py` | `LAM` (label-aware module) |
| `modules/fusion_heads.py` | `SideHead`, `FusionHead`, `DeformBlock` |
| `modules/total_loss.py` | Dice / Focal / Tversky losses, `build_loss` |
| `modules/metrics.py` | `EvalMetrics` |
| `train.py` | training loop, distillation loss, boundary GT |
| `evaluate.py` | test-split evaluation and visualisation |
| `model_complexity.py` | parameter / FLOP / FPS measurement |
| `dataset.py` | `SegDataset` — image + mask pairs at 256×256 |
| `data/registry.py` | dataset lookup, pair matching, on-demand mask conversion |
| `data/prepare.py` | COCO-polygon → mask PNG conversion |

---

## 11. Limitations worth knowing before you extend this

1. **The pipeline is binary even though the heads are not.** `LACTNet` takes a
   `num_classes` argument and the semantic heads honour it, but every loss
   (binary Dice/Focal/Tversky), the deep supervision (binary cross-entropy),
   the distillation loss (Bernoulli KL) and `EvalMetrics` assume a single
   foreground channel. Moving to multiple defect classes means changing all
   four, not just the model.
2. **The input resolution is fixed at 256×256** by the position embedding and
   the window sizes. Any larger-resolution experiment needs the position
   embedding interpolated and the Swin window scheme revisited.
3. **The FLOPs figure excludes elementwise operations, normalisation,
   interpolation and deformable-sampling overhead.** It is a conv-and-matmul
   count, so it understates the true cost — especially for the teacher, whose
   attention and deformable convolutions are both undercounted relative to a
   plain conv net.
4. **The window size is hard-coded to 7** in `SwinTransformerBlock`, while the
   stage resolutions are 64, 32, 16 and 8. The finest stage is effectively
   global attention; the coarsest is heavily padded.
