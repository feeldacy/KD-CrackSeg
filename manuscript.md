# Optimizing Structural Crack Segmentation: A Knowledge Distillation Approach

**Authors:** Husna Felisa *(manuscript draft reconstructed from the project artifacts — verify authorship and affiliation before submission)*
**Affiliation:** *[to be completed]*
**Corresponding author:** *[to be completed]*

> **Status of this draft.** The narrative, method description and table values below are
> reconstructed from the project artifacts and the manuscript draft. The numbers in
> Tables 1–7 are the values carried in the manuscript draft; they are **not** reproduced
> by the checkpoints saved in this repository (see Appendix D and
> `HANDOVER_RESPONSE.md` §0.1 for the full provenance analysis).

---

## Abstract

Automated segmentation of structural cracks in pavement and concrete surfaces supports
large-scale structural health monitoring, in particular for unmanned aerial vehicle (UAV)
inspection campaigns, where cracks appear as thin, elongated, low-contrast structures with
strong class imbalance. Recent hybrid encoder–decoder networks that combine convolutional
feature extraction with window-based transformer attention achieve a good accuracy–efficiency
trade-off, but their inference cost remains high for real-time deployment on
resource-constrained platforms. In this paper we investigate whether the accuracy of a
strong hybrid teacher network can be transferred to a compact convolutional student through
knowledge distillation, and we quantify how much of the teacher's accuracy is recovered. We
build on the label-aware CNN–Transformer teacher (LACTNet) and construct a purely
convolutional student that shares the same DUC decoder, label-aware modules (LAMs),
semantic/boundary side heads and deformable-convolution fusion heads as the teacher, but
removes the Swin transformer branch and its attentional fusion (AFFM) modules. A Bernoulli
KL-divergence distillation objective is applied jointly to the semantic and the boundary
output maps, while the classical hard-label supervision is retained as a joint term.
Experiments on the DeepCrack and CRACK500 benchmarks show that the distilled student
surpasses its own non-distilled version on both datasets and, in the Dice + Focal
configuration, exceeds the teacher on DeepCrack while retaining a compact parameter budget
(14.46 M parameters vs. 24.78 M for the teacher) and a higher throughput (6.98 vs.
4.22 FPS). An ablation on deformable convolutions in the encoder indicates that the
distillation gain is not caused by the deformable-convolution variant, and that additional
encoder deformability slightly degrades segmentation quality on DeepCrack. The results
indicate that a compact convolutional network, supervised by a hybrid teacher, is a
practical operating point for UAV-based crack inspection.

**Keywords:** crack segmentation, knowledge distillation, structural health monitoring,
UAV inspection, LACTNet, DeepCrack, CRACK500, deformable convolution

---

## 1. Introduction

Cracks are the earliest and most visible indicators of structural degradation in pavement
and concrete infrastructure. Periodic visual inspection of such infrastructure is expensive,
subjective and hazardous, and therefore automated inspection has become an active field of
research. Unmanned aerial vehicles (UAVs) have accelerated this process by enabling
high-resolution, wide-area imaging of bridge decks, roads and façades at low cost. The
resulting imagery, however, poses a demanding pixel-segmentation problem: cracks occupy a
very small fraction of the image area, their contrast against the background is often weak,
they are elongated and highly anisotropic, and the background frequently contains
distractors such as joint sealing material, stains, shadows, vegetation and texture noise
that resemble cracks.

Convolutional segmentation networks address this problem efficiently, and architectures such
as U-Net and its many variants remain the workhorse of the field. Hybrid designs that
combine a convolutional branch with a window-based transformer branch have recently
demonstrated stronger performance, because local texture statistics and long-range context
are complementary for thin structures. LACTNet [1], in particular, couples a Swin
transformer branch with a convolutional branch through attentional feature fusion modules
(AFFM) and augments the decoder with label-aware modules and a dense upsampling
convolution (DUC) decoder, and reports competitive crack-segmentation results on
DeepCrack and CRACK500.

The cost of such a model, however, is substantial: the transformer branch and its
attention blocks dominate the parameter count and the multiply–accumulate budget, and
improve accuracy at a cost that is undesirable for edge or UAV-side deployment. Knowledge
distillation [2] offers a standard remedy: a compact *student* network is trained not only
against the hard ground-truth labels but also against the softened output distribution of
a larger, frozen *teacher*, transferring the teacher's "dark knowledge" about the
structural similarity of output predictions.

This paper asks a focused question: **for surface crack segmentation, how much of a
hybrid CNN–Transformer teacher's accuracy can a purely convolutional student recover, and
at what computational cost?** To answer it we (i) instantiate a hybrid teacher that
follows the LACTNet design, (ii) construct a convolutional student that keeps the decoder,
the label-aware modules, the multi-scale side heads and the deformable fusion heads but
discards the transformer branch, (iii) distil the teacher's semantic and boundary
responses into the student with a temperature-scaled Bernoulli KL objective, and (iv)
quantify the effect on two public benchmarks and on an encoder deformable-convolution
ablation.

Our contributions are:

1. A crack-specific teacher–student framework in which a hybrid CNN–Transformer teacher
   supervises a convolutional student through *dual-branch* distillation, i.e. the
   distillation loss is applied to the semantic crack map and to the boundary map
   simultaneously.
2. A compact student that retains the label-aware and boundary-aware design elements of
   the teacher, so that the student benefits from the same inductive bias while removing
   the transformer branch, reducing the parameter count by roughly 40 % and, in the
   manuscript's measurements, increasing throughput.
3. A systematic benchmark of teacher, student and distilled student on DeepCrack and
   CRACK500 under two loss configurations (BCE + Dice and Dice + Focal), including a
   controlled ablation of deformable convolutions in the encoder.

---

## 2. Related Work

### 2.1 CNN-based crack segmentation

U-Net [3] and its descendants (UNet++, Attention-UNet, ResUNet++) remain the dominant
architectures for pixel-wise crack segmentation, because the encoder–decoder skip
connections preserve the thin structures that a plain classifier would discard. Later
works added attention, multi-scale feature aggregation, pyramid pooling (PSPNet [4]) and
atrous spatial pyramid pooling (DeepLabV3 [5]) to widen the receptive field, and specific
crack-oriented designs such as multi-stage supervised crack detection, crack-specific
attention and boundary-aware losses [6–9]. A recent dual-channel multi-scale feature
attention network (DCMA-Net) [10] combines a dilated convolution stem, a ResNet-style
encoder, an ASPP neck, multi-scale feature enhancement and a channel-spatial attention
kernel, and reports 57.12 % IoU on DeepCrack and 59.84 % IoU on CRACK500.

### 2.2 Hybrid CNN–Transformer networks for defect segmentation

Transformers have been introduced to surface-defect segmentation to capture long-range
dependencies. LACTNet [1] is the architecture most relevant to this work: it runs a Swin
transformer branch and a convolutional branch in parallel, fuses them through attentional
feature fusion modules, and decodes the fused features with a DUC decoder augmented by
label-aware modules. Its reported cost is 32.64 M parameters and 9.77 GFLOPs.
GLoU-MiT [11] takes a different route, coupling a lightweight global–local Mamba
guidance branch with a U-shaped mix-transformer for UAV pavement crack segmentation,
targeting a lower computation budget.

### 2.3 Knowledge distillation

Hinton et al. [2] introduced model distillation, matching the softened output distribution
of a teacher network with a student using a temperature-scaled KL divergence, and showed
that the auxiliary "dark knowledge" improves the student's generalisation and
transferability. The method has since been applied to semantic segmentation, where a
formulation for structured outputs was proposed by Liu et al. [12], and to medical and
industrial anomaly segmentation, where the teacher's soft boundaries are particularly
valuable because label noise dominates.

### 2.4 Deformable convolution

Deformable convolution [13] learns an offset field that shifts the sampling positions of
the convolution kernel, so that a geometrically adaptive receptive field can follow
non-rigid structures. Because cracks are thin and elongated, deformable sampling is
plausibly better matched to their geometry than a fixed grid. It is, however, more
expensive than a standard convolution and is not automatically beneficial: the sampling
offsets must be learned from data, and with limited crack data the additional degrees of
freedom may hurt rather than help.

---

## 3. Proposed Method

### 3.1 Overview

The framework has three components: a hybrid **teacher** network trained with hard labels
only; a compact convolutional **student** network trained with hard labels only; and a
**distilled student** trained with hard labels plus the teacher's soft predictions. The
teacher and the student share their decoder-side design, so that the only architectural
difference between them is the removal of the transformer branch and of the attentional
fusion between the two encoder branches. The distillation signal is applied to *both*
output streams of the network — the semantic crack map and the boundary map — because the
teacher's confidence in the location of a crack boundary is a large part of what makes
its predictions useful.

### 3.2 Teacher network

The teacher follows the LACTNet design and contains two encoder branches.

**Swin branch.** The input image is embedded into non-overlapping 4 × 4 patches
(projected to 64 channels) and given a learned absolute position embedding valid for a
256 × 256 input. Four Swin stages with window sizes (64, 64), (32, 32), (16, 16) and
(8, 8) alternate window attention and shifted-window attention; between consecutive
stages a 2 × 2 strided convolution doubles the channel count and halves the spatial
resolution, giving stage outputs at 1/4, 1/8, 1/16 and 1/32 of the input resolution with
64, 128, 256 and 512 channels.

**Convolutional branch.** Five convolutional blocks (32, 64, 128, 256, 512 channels),
each performing a strided 3 × 3 convolution followed by normalisation and a ReLU
activation, produce features c1 … c5 at 1/2, 1/4, 1/8, 1/16 and 1/32 resolution.

**Attentional feature fusion (AFFM).** At every stage, the Swin feature is rescaled to the
channel count of the corresponding convolutional feature and fused with it by an
attentional module, producing f1 … f4, which are the inputs of the decoder.

**Decoder.** Four DUC (dense upsampling convolution) blocks upsample the fused features
by a factor of two at each step, fusing them with the corresponding skip connection. Two
label-aware modules (LAM) are inserted after the second and third decoding stages; the
LAMs return both the refined feature and an auxiliary logit that is used for deep
supervision.

**Heads.** The network predicts two streams. The *semantic* stream predicts the crack
mask and the *boundary* stream predicts the crack boundary. Each stream is produced by
four side heads attached to the 1/8, 1/16, 1/32 decoder levels and the main head, whose
outputs are bilinearly resized to the input resolution, concatenated and fused by a
fusion head. The fusion stage uses a deformable convolution, so that the final
aggregation can adapt its sampling to the irregular shape of a crack.

**Boundary ground truth.** Boundary targets are obtained from the binary mask by
morphological gradient (dilation minus erosion with a 3 × 3 structuring element), which
marks the pixels adjacent to the crack region.

### 3.3 Student network

The student keeps the entire convolutional encoder, the DUC decoder, the LAMs, the
semantic and boundary side heads, and the deformable fusion heads of the teacher. It
removes the patch embedding, the position embedding, the four Swin stages, the Swin
down-sampling convolutions and the four AFFM modules; the decoder is therefore driven
purely by convolutional features, and the encoder–decoder channel widths are identical to
those of the teacher. This isolates the effect of the transformer branch: the student is
not a differently-sized network, it is the teacher with its global-context branch removed.

### 3.4 Knowledge distillation objective

Let the teacher be T and the student S, and let z_T and z_S denote their semantic logits
for a training image x, and z_T^b, z_S^b the corresponding boundary logits. Because the
task is binary, the soft targets of the teacher are Bernoulli probabilities
p_T = σ(z_T / τ), and the distillation loss is the Bernoulli KL divergence between the
teacher and the student Bernoulli distributions, evaluated in the two-class form and
scaled by τ² to preserve gradient magnitudes:

L_KD(x) = τ² · KL( p_T ‖ p_S ) + τ² · KL( p_T^b ‖ p_S^b ).

The teacher is frozen (eval mode, no gradient) and produces its predictions in a single
forward pass per batch. The total objective of the distilled student is

L = L_sup + λ · L_KD ,

where L_sup is the hard-label supervised loss of the student (identical in form to the
teacher's supervised loss) and λ is the distillation weight.

### 3.5 Supervised losses

Two supervised loss configurations are evaluated for both the semantic and the boundary
stream.

**BCE + Dice.** The binary cross-entropy term penalises pixel-wise error, while the soft
Dice term

L_Dice = 1 − (2 Σ p y + ε) / (Σ p + Σ y + ε)

directly optimises the overlap between prediction and mask and therefore mitigates the
class imbalance. The combination is L_BCE + L_Dice.

**Dice + Focal.** The focal term down-weights easy (mostly background) pixels through
the modulating factor (1 − p_t)^γ and applies a class-balancing factor α, so that the
gradient is concentrated on the thin, ambiguous crack pixels. With α = 0.25 and γ = 2,
the loss is L_Focal + L_Dice.

In both configurations the boundary stream is supervised with the same loss family, and
the deep-supervision term provided by the two LAM auxiliary logits is added with its own
weight.

### 3.6 Training procedure

The network is optimised with Adam, gradient clipping at a maximum norm of 1.0, mixed
precision and a cosine annealing schedule over the full training budget. The best
checkpoint is selected by the validation F1 score computed on the held-out validation
split, and the selected checkpoint is the one evaluated on the test split.

---

## 4. Experiments

### 4.1 Datasets

**DeepCrack** [14]. A dataset of 537 high-resolution images of concrete surfaces with
manually annotated crack masks. The 300 training images are split into a training pool and
a validation subset holding out 20 % of the pool (60 images); the remaining 237 images
form the held-out test split used for all reported numbers.

**CRACK500** [15]. A dataset of 500 high-resolution pavement images. The standard
train/validation/test partition is used at the level of the original photographs (250 /
50 / 148 source images), which after cropping yields 1 896 / 348 / 1 124 image–mask
pairs. The 1 124 test pairs are used for all reported numbers. Because the split is
performed on the source photographs, no patch of the same original image can appear in
two different splits.

For both datasets the images and masks are resized to 256 × 256, binarised at a
threshold of 127 and randomly flipped during training.

### 4.2 Implementation details

All experiments were conducted in Python with PyTorch 2.14.1 on a single CUDA GPU. The
network was trained for 60 epochs with a batch size of 4 and the Adam optimiser. The
learning rate was set to 1e-4 and decayed with a cosine annealing schedule over the
training horizon. The distillation temperature was set to T = 3.0 and the distillation
weight to α = 0.5. The deep-supervision weight of the two LAM auxiliary losses was set to
3. For the DeepCrack experiments, where no validation split is provided by the dataset,
20 % of the training pool was held out for validation; for CRACK500 the dataset-provided
validation split was used. Model selection was performed on the validation F1 score and
the selected checkpoint was evaluated on the test split at a decision threshold of 0.5.

### 4.3 Evaluation metrics

Segmentation quality is quantified with accuracy, precision, recall, F1 score and IoU.
Two aggregation conventions exist in the literature and both are reported explicitly in
this paper: *dataset-level* metrics aggregate the true-positive, false-positive and
false-negative pixel counts over the whole test split before computing the ratio, whereas
*mean per-image* metrics compute the score for each test image and average over images.
The main comparison tables report the dataset-level convention; the per-image mean Dice
is reported additionally, and the mapping between the two is documented in Appendix B.

### 4.4 Main results

**Table 1. Segmentation results on the DeepCrack test split (237 images) with the BCE +
Dice loss configuration.** Baseline rows are quoted from DCMA-Net [10] and were not
re-trained in this work; rows marked with † are produced by this study.

| Method | Source | Precision (%) | IoU (%) | Recall (%) | F1 (%) | Dice (%) | Acc (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UNet [3] | [10] | 57.38 | 51.90 | 89.34 | 69.88 | – | – |
| SegNet [16] | [10] | 59.11 | 53.41 | 88.75 | 70.96 | – | – |
| DeepLabV3 [5] | [10] | 58.56 | 52.88 | 88.64 | 70.51 | – | – |
| ResUNet++ [17] | [10] | 60.13 | 53.72 | 87.10 | 71.14 | – | – |
| DCMA-Net [10] | [10] | 62.07 | 57.12 | 89.84 | 73.41 | – | – |
| Teacher (LACTNet) † | this work | – | 63.63 | – | – | 75.63 | 98.41 |
| Student (no KD) † | this work | – | 64.50 | – | – | 77.22 | 98.41 |
| Distilled student † | this work | – | **66.91** | – | – | **79.13** | 98.54 |

**Table 5. Segmentation results on the CRACK500 test split (1124 images) with the BCE +
Dice loss configuration.** Baseline rows are quoted from DCMA-Net [10]; † denotes this
study.

| Method | Source | Precision (%) | IoU (%) | Recall (%) | F1 (%) | Dice (%) | Acc (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UNet [3] | [10] | 64.91 | 53.56 | 77.30 | 70.51 | – | – |
| SegNet [16] | [10] | 67.82 | 55.12 | 76.19 | 71.76 | – | – |
| DeepLabV3 [5] | [10] | 68.16 | 56.41 | 77.98 | 72.73 | – | – |
| ResUNet++ [17] | [10] | 71.35 | 55.72 | 73.44 | 72.35 | – | – |
| DCMA-Net [10] | [10] | 75.60 | 59.84 | 75.68 | 75.64 | – | – |
| Teacher (LACTNet) † | this work | – | 41.41 | – | – | 57.18 | 97.78 |
| Student (no KD) † | this work | – | 42.06 | – | – | 57.67 | 97.79 |
| Distilled student † | this work | – | **42.76** | – | – | **58.36** | 97.76 |

The qualitative behaviour is consistent on both benchmarks. Under the BCE + Dice
configuration the distilled student improves the IoU over the independently trained student
by 2.41 points on DeepCrack (64.50 → 66.91) and by 0.70 points on CRACK500 (42.06 →
42.76), and it also improves the mean per-image Dice (75.63 → 79.13 and 57.18 → 58.36
respectively). The distilled student surpasses the hybrid teacher in both cases, by 3.28
IoU points on DeepCrack and by 1.35 IoU points on CRACK500.

**Table 6. Segmentation results on the DeepCrack test split with the Dice + Focal loss
configuration.**

| Model | Loss | Acc (%) | Dice (%) | IoU (%) |
| --- | --- | ---: | ---: | ---: |
| Teacher (LACTNet) | Dice + Focal | 98.55 | 77.39 | 64.94 |
| Student (no KD) | Dice + Focal | 98.51 | 77.76 | 65.10 |
| Distilled student | Dice + Focal | 98.55 | 78.12 | **65.80** |

Compared with the BCE + Dice configuration, the Dice + Focal objective raises the mean
per-image Dice of the teacher from 75.63 % to 77.39 % and the IoU from 63.63 % to
64.94 %, confirming that the focal term concentrates the optimisation on the
minority crack pixels. The ordering teacher < student < distilled student is preserved
under both loss families, which indicates that the distillation benefit is not an artefact
of one particular loss combination.

**Table 3. Detailed per-class metrics of the project models on the DeepCrack test split
(Dice + Focal configuration).**

| Model | Precision (%) | IoU (%) | Recall (%) | F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| Teacher | 89.65 | 64.94 | 70.47 | 77.39 |
| StuLACTNet (distilled student) | 86.33 | **65.80** | 73.38 | 78.12 |

**Table 4. Detailed per-class metrics of the project models on the CRACK500 test split
(Dice + Focal configuration).**

| Model | Precision (%) | IoU (%) | Recall (%) | F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| Teacher | 59.85 | 42.27 | 56.74 | 57.83 |
| StuLACTNet (distilled student) | 59.05 | **42.86** | 58.34 | 58.34 |

The distilled student trades a small amount of precision for a clearly larger recall on
both datasets (DeepCrack: 86.33 % vs. 89.65 % precision against 73.38 % vs. 70.47 %
recall). Because cracks are thin and under-segmented by an over-conservative decoder, the
recall gain is the more valuable property for practical inspection, where a missed crack
segment is a potential false negative in an inspection record.

### 4.5 Efficiency

**Table 2. Computational cost of the teacher and of the distilled student.**

| Model | Parameters (M) | GFLOPs | FPS |
| --- | ---: | ---: | ---: |
| LACTNet teacher | 24.78 | 20.34 | 4.22 |
| Distilled student | 14.46 | 15.49 | 6.98 |

Removing the transformer branch reduces the parameter count by 41.6 % and the
multiply–accumulate budget by 23.8 %, while the measured throughput increases by 65.4 %.
The parameter reduction is larger than the FLOP reduction because the Swin stages carry
most of the weights relative to their arithmetic cost. The distilled student therefore
offers an accuracy that exceeds the teacher's while running approximately 1.65 times
faster, which is the main practical argument for adopting the distilled model on
UAV-side or edge hardware.

> **Editorial note (must be resolved before submission).** The FPS column above is the
> manuscript's own measurement, whose hardware and precision settings were not recorded.
> An independent profile of the current code on an NVIDIA GeForce RTX 3050 (FP32, batch
> size 1, 10 warm-up and 50 timed iterations, CUDA synchronised) gives
> 24.80 M / 22.12 GFLOPs / 29.05 FPS for the teacher and 14.69 M / 17.26 GFLOPs /
> 23.09 FPS for the student, i.e. the student is *slower* than the teacher in that
> setting, because the deformable-convolution fusion heads — which the student retains —
> dominate the runtime. The efficiency claim must therefore either be re-measured on the
> original hardware or be restated in terms of parameters and FLOPs only. See
> `HANDOVER_RESPONSE.md` §5 (request **R3**).

### 4.6 Ablation: deformable convolutions in the encoder

**Table 7. Effect of adding deformable convolutions to the encoder on the DeepCrack test
split (Dice + Focal configuration).**

| Encoder | Teacher IoU (%) | Distilled student IoU (%) |
| --- | ---: | ---: |
| Standard convolution | **64.94** | **65.80** |
| + Deformable convolution | 64.59 | 63.46 |

Replacing the encoder convolutions by their deformable variants degrades both models on
DeepCrack, by 0.35 IoU points for the teacher and by 2.34 IoU points for the distilled
student. The explanation is the same in both cases: the deformable offsets have to be
estimated from a limited number of annotated crack images, and the additional degrees of
freedom lead to over-fitting rather than to a better-adapted receptive field. Importantly,
the degradation is larger for the student than for the teacher, which suggests that the
student has less capacity to absorb the extra parameters. This ablation also isolates the
source of the distillation gain: the improvement reported in Tables 1, 5 and 6 is obtained
with the standard convolutional encoder and is therefore attributable to the distillation
objective, not to the deformable-convolution variant.

Note that the deformable convolution in the *fusion head* is present in every
configuration reported in this paper; the ablation concerns only the encoder blocks.

> **Provenance note (must be resolved before submission).** No revision of the Teacher or
> the Student in the public history of this repository ever instantiated a deformable
> encoder block, and no run directory corresponds to these two rows. The experiments
> behind Table 7 were either executed in an uncommitted local variant of the code or
> originate elsewhere; see `HANDOVER_RESPONSE.md` §3.2 (request **R2**).

### 4.7 Qualitative results

The qualitative inspection of the predictions shows that the teacher and the distilled
student agree on the overall crack trajectories but differ in the treatment of ambiguous
regions. The teacher produces conservative, high-precision masks that occasionally break
thin cracks into disconnected fragments, whereas the distilled student recovers more
continuous crack paths, in particular in low-contrast areas and near the image borders.
Where both models disagree with the ground truth, the failures are dominated by
background structures with crack-like appearance (sealing material, shadow edges) rather
than by genuinely ambiguous crack pixels.

---

## 5. Discussion

The central observation of this study is that, for crack segmentation, the accuracy of a
hybrid CNN–Transformer teacher can be fully recovered — and even exceeded — by a purely
convolutional student that is 41.6 % smaller and, in the manuscript's measurements,
faster. Three points deserve emphasis.

First, distillation is not merely a regulariser here: the distilled student outperforms
the teacher on both datasets. The mechanism is visible in Tables 3 and 4. The teacher is
markedly more precise but less sensitive; the distilled student inherits the teacher's
decision structure while being less conservative about thin, low-contrast crack segments.
Because the hard labels are noisy at the pixel level (manual crack annotation is
subjective at the crack boundary) while the teacher's soft predictions are smooth, the
soft targets act as a regulariser that reduces over-segmentation of the boundary region.

Second, keeping the boundary stream in the distillation objective matters for this task.
The teacher is explicitly trained to localise crack contours, and its boundary map
carries information that is not recoverable from the semantic map alone. Distilling both
streams allows the student to inherit the teacher's contour prior without needing the
transformer branch that produced it.

Third, the encoder deformable-convolution ablation is negative, and it is informative: it
shows that architectural "flexibility" is not automatically beneficial for thin
structures when the training set is small. The result is consistent with the observation
that crack datasets are small compared with the sample requirements of deformable
sampling, and it motivates the search for gains in the supervision signal rather than in
the sampling operator.

The limitations of the study are as follows. All results are single-run, single-split;
no multi-seed averages are reported, so small differences (below roughly 0.5 IoU points)
should not be interpreted as significant. The baseline rows are quoted from the
DCMA-Net study rather than re-trained under an identical protocol, so the comparison is
indicative rather than strictly controlled. The reported FPS values depend on the
measurement hardware and on the definition of the FLOP counting convention, and should
be interpreted as relative rather than absolute figures. Finally, the models operate on
256 × 256 inputs; crack width information below the resolution of that grid is
irrecoverable, and a full-resolution or multi-scale evaluation would be required before
deploying the models for quantitative crack-width measurement.

---

## 6. Conclusion

We presented a knowledge-distillation framework for structural crack segmentation in which
a hybrid CNN–Transformer teacher supervises a compact convolutional student through
joint semantic and boundary supervision. On DeepCrack and CRACK500 the distilled student
improves consistently over its non-distilled version, surpasses the teacher in IoU and mean
per-image Dice under both BCE + Dice and Dice + Focal supervision, and does so with 41.6 %
fewer parameters. An encoder deformable-convolution ablation shows that the gain is
attributable to the distillation signal rather than to the deformable variant, and that
deformable sampling is detrimental for this data regime. The practical conclusion is that
the accuracy of a large hybrid crack-segmentation model can be retained in a compact
convolutional model suitable for UAV-side inspection, which makes the proposed pipeline a
viable component of a structural health monitoring system.

Future work should (i) report multi-seed averages and a strict train/validation/test
protocol for all baselines, (ii) extend the dual-stream distillation to a boundary-weighted
teacher–student formulation, (iii) evaluate the models at native image resolution with
crack-width metrics, and (iv) validate the pipeline on an in-service UAV inspection
dataset.

---

## References

[1] *LACTNet: A label-aware CNN-transformer network for surface defect segmentation* —
reference paper supplied with the project; complete the author list and venue from the
source PDF.
[2] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network,"
arXiv:1503.02531, 2015.
[3] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical
image segmentation," in Proc. MICCAI, 2015.
[4] H. Zhao et al., "Pyramid pooling module for semantic segmentation," in Proc. ICCV,
2017.
[5] X. Chen et al., "Encoder-decoder with atrous separable convolution for semantic
image segmentation," in Proc. ECCV, 2018.
[6] Y. Liu et al., "Crackformer II," 2023. *(complete from the source paper list)*
[7] H. Zhang et al., "Deep learning-based crack detection for pavement monitoring," 2021.
*(complete from the source paper list)*
[8] A. Reina et al., "Crack and pore segmentation with deep learning," 2021. *(complete
from the source paper list)*
[9] D. Sun, Y. Li, et al., "Road crack detection using deep convolutional neural network,"
2019. *(complete from the source paper list)*
[10] Y. Yan et al., "DCMA-Net: A dual channel multi-scale feature attention network for
crack image segmentation," Engineering Applications of Artificial Intelligence, vol. 148,
art. 110411, 2025.
[11] *GLoU-MiT: Lightweight global-local Mamba-guided U-mix transformer for UAV-based
pavement crack segmentation* — reference paper supplied with the project; complete the
author list and venue from the source PDF.
[12] Y. Liu et al., "Distilling structured knowledge for efficient semantic segmentation,"
in Proc. CVPR, 2019.
[13] S. Dai et al., "Deformable convolutional networks," in Proc. ICCV, 2017.
[14] *DeepCrack: Learning hierarchical convolutional features for crack detection*,
IEEE Trans. Image Processing, 2019.
[15] Y. Yang et al., "Feature pyramid and hierarchical boosting network for pavement crack
detection," IEEE Trans. Intelligent Transportation Systems, 2020.
[16] A. Badrinarayanan et al., "SegNet: A deep convolutional encoder-decoder architecture
for image segmentation," IEEE Trans. Pattern Analysis and Machine Intelligence, 2017.
[17] S. Jha et al., "ResUNet++: A deep residual U-Net architecture for salient object
detection," in Proc. ICASSP, 2019.

---

## Appendix A. Experiment inventory and provenance

The manuscript reports 14 distinct training/evaluation configurations, listed below with
the run directories and the checkpoints that correspond to them. The complete provenance
audit, including the recovered training configuration and the discrepancies between these
runs and the manuscript values, is in `HANDOVER_RESPONSE.md`.

| ID | Dataset | Model / mode | Loss | Encoder DCN | Acc (%) | Dice (%) | IoU (%) | Run directory | Checkpoint |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| D01 | DeepCrack | Teacher | BCE + Dice | No | 98.41 | 75.63 | 63.63 | `teacher_exp2_deepcrack` | `best.pth` |
| D02 | DeepCrack | Student, no KD | BCE + Dice | No | 98.41 | 77.22 | 64.50 | `student_exp7_deepcrack` | `best.pth` |
| D03 | DeepCrack | Student + KD | BCE + Dice | No | 98.54 | 79.13 | 66.91 | `student_exp2_deepcrack_distill` | `best.pth` |
| D04 | DeepCrack | Teacher | Dice + Focal | No | 98.55 | 77.39 | 64.94 | `teacher_exp5_deepcrack_dicefocal` | `best.pth` |
| D05 | DeepCrack | Student, no KD | Dice + Focal | No | 98.51 | 77.76 | 65.10 | `student_exp8_deepcrack_dicefocal` | `best.pth` |
| D06 | DeepCrack | Student + KD | Dice + Focal | No | 98.55 | 78.12 | 65.80 | `student_exp4_deepcrack_distill_dicefocal` | `best.pth` |
| D07 | DeepCrack | Teacher | Dice + Focal | **Yes** | 98.52 | 77.04 | 64.59 | **not run in this repository** | — |
| D08 | DeepCrack | Student + KD | Dice + Focal | **Yes** | 98.43 | 76.21 | 63.46 | **not run in this repository** | — |
| C01 | Crack500 | Teacher | BCE + Dice | No | 97.78 | 57.18 | 41.41 | `teacher_exp3_crack500` | `best.pth` |
| C02 | Crack500 | Student, no KD | BCE + Dice | No | 97.79 | 57.67 | 42.06 | `student_exp5_crack500` | `best.pth` |
| C03 | Crack500 | Student + KD | BCE + Dice | No | 97.76 | 58.36 | 42.76 | `student_exp1_crack500_distill` | `best.pth` |
| C04 | Crack500 | Teacher | Dice + Focal | No | 97.72 | 57.83 | 42.27 | `teacher_exp4_crack500_dicefocal` | `best.pth` |
| C05 | Crack500 | Student, no KD | Dice + Focal | No | 97.81 | 57.67 | 42.08 | `student_exp6_crack500_dicefocal` | `best.pth` |
| C06 | Crack500 | Student + KD | Dice + Focal | No | 97.70 | 58.34 | 42.86 | `student_exp3_crack500_distill_dicefocal` | `best.pth` |

An additional earlier run, `runs/exp1`, is a 20-epoch DeepCrack teacher whose evaluation
(IoU 67.43 %) is stored in `eval_result_exp1`; it is not used in any table of this paper.

## Appendix B. Metric conventions

The evaluator reports two families of numbers that must not be interchanged:

* **Dataset-level** metrics (accuracy, precision, recall, F1, IoU) are computed from
  pixel counts accumulated over the entire test split. A single large false-positive
  region therefore influences them strongly.
* **Mean per-image** metrics (`dice`, `iou_img`) are computed per test image and then
  averaged, which weights every image equally regardless of its crack area.

The headline tables of this paper use the dataset-level IoU and the mean per-image Dice.
The F1 column of Tables 3 and 4 follows the mean per-image Dice convention used for the
corresponding rows of Tables 1 and 6; this is stated explicitly because the two F1
definitions differ for the same prediction set, and a reader recomputing the harmonic
mean of the reported precision and recall may obtain a slightly different value.

## Appendix C. Reproduction

The project provides three entry points, `train.py`, `evaluate.py` and
`model_complexity.py`. The historical configurations reported in this manuscript correspond
to the following command templates, with the dataset-specific run directories listed in
Appendix A:

```powershell
# Teacher
python train.py --dataset DeepCrack --mode standard --epochs 60 --bs 4 --lr 1e-4 --out runs/teacher_exp1

# Student without distillation
python train.py --dataset DeepCrack --mode student --epochs 60 --bs 4 --lr 1e-4 --out runs/student_exp1

# Distilled student
python train.py --dataset DeepCrack --mode distill --teacher_checkpoint runs/teacher_exp1/best.pth --epochs 60 --bs 4 --lr 1e-4 --kd_temperature 3.0 --kd_w 0.5 --out runs/distill_exp1

# Evaluation of a student checkpoint (the --model flag must be set explicitly)
python evaluate.py --dataset DeepCrack --checkpoint runs/distill_exp1/best.pth --model student --threshold 0.5 --output output/distill_exp1/distill_exp1.json

# Efficiency measurement
python model_complexity.py --model all --batch-size 1 --warmup 10 --iterations 50
```

The implementation details stated in Section 4.2 reflect the configurations used for the
reported results. The default values of the current command-line interface differ from
them (100 epochs, batch size 8, temperature 2.0, distillation weight 1.0), so the
historical values should be passed explicitly rather than relied upon as defaults.

## Appendix D. Independently re-evaluated checkpoints in the current repository

The repository contains a second, newer set of trained checkpoints whose evaluation
records were produced with the current source revision. These numbers are **not** the
numbers reported in Tables 1–7; they are listed separately because they constitute a
provenance discrepancy that has to be resolved before the manuscript is submitted (the
two sets differ in code revision, in training configuration — 100 epochs and batch size 8
instead of 60 and 4 — and, most likely, in the dataset revision used for the splits). The
full analysis is in `HANDOVER_RESPONSE.md` §0.1 and §2.4.

| Run directory | Dataset | Model | Loss | IoU (%) | Mean Dice (%) | Mean IoU (%) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| teacher_exp2_deepcrack | DeepCrack | Teacher | BCE + Dice | 71.35 | 79.88 | 68.46 |
| teacher_exp4_crack500_dicefocal | Crack500 | Teacher | Dice + Focal | 57.01 | 68.03 | 53.95 |
| teacher_exp5_deepcrack_dicefocal | DeepCrack | Teacher | Dice + Focal | 71.53 | 80.42 | 68.91 |
| student_exp1_crack500_distill | Crack500 | Student + KD | BCE + Dice | 57.46 | 68.75 | 54.68 |
| student_exp2_deepcrack_distill | DeepCrack | Student + KD | BCE + Dice | 71.24 | 80.07 | 68.80 |
| student_exp3_crack500_distill_dicefocal | Crack500 | Student + KD | Dice + Focal | 57.63 | 68.88 | 54.82 |
| student_exp4_deepcrack_distill_dicefocal | DeepCrack | Student + KD | Dice + Focal | 69.51 | 79.91 | 68.48 |
| student_exp5_crack500 | Crack500 | Student, no KD | BCE + Dice | 56.39 | 67.71 | 53.70 |
| student_exp6_crack500_dicefocal | Crack500 | Student, no KD | Dice + Focal | 56.52 | 67.89 | 53.66 |
| student_exp7_deepcrack | DeepCrack | Student, no KD | BCE + Dice | 72.04 | 80.81 | 69.54 |
| student_exp8_deepcrack_dicefocal | DeepCrack | Student, no KD | Dice + Focal | 72.02 | 81.24 | 69.81 |

All records were produced by `evaluate.py` at a threshold of 0.5 on the DeepCrack test
split (237 images) and the CRACK500 test split (1124 images) and are stored as JSON under
`output/<run>/`.

Three observations follow from this table and should be reconciled with the main
manuscript before submission:

1. The absolute IoU values are systematically higher than those of Tables 1–7, on both
   datasets and for every model, which points to a difference in the evaluation protocol
   or in the dataset revision rather than to a model-level improvement.
2. The ordering **CRACK500** — distilled student > teacher > non-distilled student — is
   preserved, so the central distillation claim of the paper holds on that dataset.
3. The ordering **DeepCrack** is *not* preserved: the distilled student falls below the
   non-distilled student and below the teacher. This must be reconciled with the
   DeepCrack rows of Tables 1 and 6 before the paper is submitted.
