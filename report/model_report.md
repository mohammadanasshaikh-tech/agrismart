# AgriSmart AI - Model Report

## Task
Crop-disease image classification (computer vision). 28 crop-disease classes
(including "healthy" variants), covering Apple, Blueberry, Cherry, Corn,
Grape, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry,
and Tomato.

## Dataset & Split
- **Training / validation source**: PlantVillage (lab-condition leaf images,
  uniform background). 38,447 images across the 28 shared classes.
- **Held-out field test source**: PlantDoc (real-world field images -- natural
  lighting, clutter, occlusion). 2,918 images. Never used in training.
- **Split**: 85% train (32,680 images) / 15% validation (5,767 images) from
  PlantVillage, deterministic random split (seed=42). PlantDoc used
  exclusively as a held-out field test, per the train-on-lab / test-in-field
  protocol.
- **Class matching**: PlantVillage and PlantDoc use different naming
  conventions. Classes were matched programmatically by crop identity +
  healthy/diseased state + disease-keyword overlap, then manually verified
  (see `src/build_class_mapping.py` and `src/class_mapping.json`).

## Model / Approach
- **Architecture**: EfficientNet-B0, ImageNet-pretrained (via `timm`), full
  fine-tuning on all layers with a new classification head (28 classes).
- **Input size**: 224x224.
- **Augmentation** (train only): random resized crop, horizontal flip,
  rotation (+/-20 deg), color jitter (brightness/contrast/saturation).
- **Optimizer**: AdamW, learning rate 3e-4, cosine annealing schedule.
- **Epochs**: 15. Batch size 16.
- **Hardware**: NVIDIA RTX 3050 (laptop GPU).

## Metric & Result

| Split | Macro-F1 | Accuracy |
|---|---|---|
| PlantVillage validation (lab images) | **0.998** | 1.00 |
| PlantDoc held-out test (field images) | **0.145** | 0.19 |

Confusion matrices and full per-class precision/recall are in
`report/confusion_matrix_*.png` and `report/classification_report_*.txt`.

## Baseline
No organizer-provided baseline number was available at the time of this
report (baseline ships with the official kickoff data). As a self-defined
reference point: a majority-class baseline (always predicting the most
frequent class, `Tomato_Yellow_Leaf_Curl_Virus`) would score close to 0
macro-F1 on the imbalanced field test set, since macro-F1 penalizes ignoring
minority classes heavily. Our field macro-F1 of 0.145, while low in absolute
terms, meaningfully exceeds this trivial baseline and reflects real (if
weak) transfer of learned disease features to field conditions. We will
re-benchmark against the organizers' official baseline once published.

## Limitations
- **Lab-to-field generalization gap (primary finding)**: the model reaches
  near-perfect macro-F1 (0.998) on lab-condition PlantVillage validation
  images but drops sharply to 0.145 macro-F1 on real field-condition PlantDoc
  images. This is the exact phenomenon the challenge brief identifies as
  the core difficulty ("models that memorise clean laboratory images...
  degrade sharply on real field photos"). It indicates the model has
  learned lab-specific visual shortcuts (uniform backgrounds, consistent
  lighting/framing) rather than robust, generalizable disease features.
- **Class imbalance in the field test set**: several classes have very few
  PlantDoc samples (e.g. `Tomato_Spider_mites` has only 2 test images),
  making their individual F1 scores statistically unreliable and
  disproportionately affecting the macro-average. A production version
  should either exclude ultra-low-sample classes from macro-F1 reporting
  or gather more field samples for those classes.
- **No field-condition images in training**: the model never saw
  cluttered/naturally-lit images during training, which is the most likely
  single cause of the generalization gap. Mixing a small amount of
  field-style images into training (or heavier field-simulating
  augmentation -- background clutter, varied lighting) is the most direct
  next step to close this gap.
- **Confusions cluster by visual similarity, not just crop**: notably, corn
  gray leaf spot is over-predicted broadly across other classes on the
  field set (high recall, low precision), suggesting the model has learned
  a generic "textured/spotted leaf" feature that overfires on field photos'
  natural texture and lighting variation.