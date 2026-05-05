# medclip-reproduced

> Clean reproduction of **MedCLIP** (Wang et al., EMNLP 2022) - contrastive learning from unpaired medical images and text for zero‑shot classification, supervised learning, and image-text retrieval.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Paper: [MedCLIP: Contrastive Learning from Unpaired Medical Images and Text](https://aclanthology.org/2022.emnlp-main.256/) - Wang et al., EMNLP 2022  
Official repo: <https://github.com/RyanWangZf/MedCLIP>

## What this is

MedCLIP is a vision–language framework for medical imaging that **decouples images and texts** during contrastive pre‑training, enabling combinatorial scaling of training data with low cost. It replaces the standard InfoNCE loss with a **semantic matching loss** based on medical knowledge to eliminate false negatives (e.g., images and reports from different patients that share the same pathology but are incorrectly treated as negatives).

This is the **sixth reproduction** in a series of visual anomaly detection I'm building toward **UniVAD** (CVPR 2025). MedCLIP is a foundational medical vision–language model that later anomaly detection methods (including UniVAD) rely on for cross‑modal understanding.

## Status

In development. Code and configuration will land incrementally.

## Why MedCLIP

Medical image–text datasets are **orders of magnitude smaller** than general‑domain ones (e.g., MIMIC‑CXR: ~377k images vs. CLIP’s 400M). Standard CLIP suffers from:

- **Data hunger** – not enough paired examples.
- **False negatives** – two images with the same pathology but from different patients are treated as negatives, slowing learning.

MedCLIP solves both with a simple, data‑efficient design. Surprisingly, it **outperforms prior SOTA even with only 20k pre‑training samples**, whereas others need ~200k.

To achieve this, MedCLIP introduces two key ideas:

1. **Decoupled training data** - Instead of requiring paired (image, report) tuples, MedCLIP treats all images and all texts as independent samples. Positive pairs are created dynamically when an image and a text share at least one common medical concept (e.g., “cardiomegaly”). This turns \(N\) images and \(M\) texts into up to \(N \times M\) viable positive pairs.

2. **Semantic matching loss** - Replaces InfoNCE with a margin‑based loss that only pushes apart hard negatives (texts sharing no concept with the image), guided by medical domain knowledge. This reduces noisy gradients from false negatives.

## Goal

Reproduce the core results reported in the MedCLIP paper on chest X‑ray datasets:

| Task                               | Dataset          | Metric           | Paper (MedCLIP) | This Repo |
| ---------------------------------- | ---------------- | ---------------- | --------------- | --------- |
| **Zero‑shot classification**       | CheXpert‑5x200   | AUROC (macro)    | 85.2            | TBD       |
|                                    | MIMIC‑5x200      | AUROC (macro)    | 81.6            | TBD       |
|                                    | COVID‑19         | AUROC (macro)    | 92.4            | TBD       |
|                                    | RSNA Pneumonia   | AUROC (macro)    | 89.1            | TBD       |
| **Supervised classification**      | CheXpert‑5x200 (1% labeled) | AUROC (macro) | 83.5 | TBD |
| **Image‑Text Retrieval**           | CheXpert‑5x200   | R@10 (Image→Text)| 34.2            | TBD       |

Per‑dataset tables will land here as runs complete.

## Installation

Clone and create the environment:

    git clone https://github.com/hammadhaideer/medclip-reproduced.git
    cd medclip-reproduced
    conda env create -f environment.yml
    conda activate medclip

Or with pip in an existing environment:

    pip install -r requirements.txt

## Dataset

### MIMIC‑CXR (main pre‑training and evaluation)

1. Request access from [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/) (requires CITI training).
2. Download the **MIMIC‑CXR JPG** dataset (≈ 300 GB).
3. Extract and ensure the following structure:

    MIMIC‑CXR‑JPG/
    ├── train/
    │   └── p10/
    │       └── p10000001/
    │           ├── s50414267/
    │           │   ├── 02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
    │           │   └── ...
    │           └── ...
    └── test/
        └── ...

4. Set an environment variable:

    export MIMIC_CXR_ROOT=/path/to/MIMIC‑CXR‑JPG

5. Download the pre‑computed CheXpert label mapping from the [official CheXpert labeler](https://github.com/stanfordmlgroup/chexpert-labeler) to generate sentence‑level semantic labels (see `data/README.md` for details).

### CheXpert (alternative for evaluation only)

If MIMIC‑CXR is too large, you can use **CheXpert** (≈ 20 GB) for zero‑shot evaluation. Set `CHEXPERT_ROOT` similarly.

| Dataset   | Source                                                                 | Size (approx) |
| --------- | ---------------------------------------------------------------------- | ------------- |
| MIMIC‑CXR | https://physionet.org/content/mimic-cxr/2.0.0/                         | 300 GB        |
| CheXpert  | https://stanfordmlgroup.github.io/competitions/chexpert/               | 20 GB         |

## Run

Pre‑train MedCLIP from scratch on MIMIC‑CXR:

    python scripts/train.py --config configs/default.yaml

Key parameters in `configs/default.yaml`:

| Parameter           | Value                     | Description                              |
| ------------------- | ------------------------- | ---------------------------------------- |
| `image_encoder`     | `resnet50` or `vit_base_patch16` | Vision backbone                     |
| `text_encoder`      | `bert-base-uncased`       | Text encoder                             |
| `batch_size`        | 64                        | Training batch size                      |
| `learning_rate`     | 1e-4                      | Initial LR                               |
| `margin`            | 0.2                       | Margin for semantic matching loss        |
| `concept_threshold` | 0.5                       | Threshold for building positive pairs    |

### Zero‑shot classification

    python scripts/eval_zero_shot.py \
        --checkpoint checkpoints/medclip_best.pth \
        --dataset chexpert \
        --split test \
        --out results/zeroshot_chexpert.json

### Supervised classification (linear probe)

    python scripts/eval_supervised.py \
        --checkpoint checkpoints/medclip_best.pth \
        --dataset mimic \
        --label_frac 0.01 \
        --out results/supervised_mimic_1pct.json

### Image‑text retrieval

    python scripts/eval_retrieval.py \
        --checkpoint checkpoints/medclip_best.pth \
        --dataset chexpert \
        --out results/retrieval_chexpert.json

### Aggregate results

    python scripts/aggregate_results.py --results_dir results/

Per‑dataset results land in `results/*.json`. The aggregated table prints to stdout, comparing your numbers against the paper's.

## Walkthrough notebook

For an interactive end‑to‑end pipeline - data sampling, semantic matching, loss computation, and zero‑shot inference:

    jupyter notebook notebooks/01_walkthrough.ipynb

Run cells top‑to‑bottom. The notebook reads the same config and dataset paths as the scripts.

## Roadmap

- [ ] Repo scaffold, configs, dataset loader
- [ ] ResNet50 / ViT image encoder
- [ ] Text encoder (BioBERT / ClinicalBERT)
- [ ] Decoupled data sampling with semantic concept matching
- [ ] Semantic matching loss implementation
- [ ] Zero‑shot classification evaluation
- [ ] Supervised (linear probe) evaluation
- [ ] Image‑text retrieval evaluation
- [ ] Training loop with AMP, gradient accumulation
- [ ] Walkthrough notebook with concept matching visualization
- [ ] Full reproduction across CheXpert, MIMIC‑5x200, COVID, RSNA
- [ ] Medium walkthrough post

## Reproduction series

Part of a broader series reproducing UniVAD's full comparison set:

- [x] [`patchcore-reproduced`](https://github.com/hammadhaideer/patchcore-reproduced) – PatchCore (CVPR 2022)
- [x] [`winclip-reproduced`](https://github.com/hammadhaideer/winclip-reproduced) – WinCLIP (CVPR 2023)
- [ ] [`uniad-reproduced`](https://github.com/hammadhaideer/uniad-reproduced) – UniAD (NeurIPS 2022)
- [ ] `anomalygpt-reproduced` – AnomalyGPT (AAAI 2024)
- [ ] `comad-reproduced` – ComAD (PR 2024)
- [ ] `medclip-reproduced` – MedCLIP (EMNLP 2022) ← this repo
- [ ] `univad-reproduced` – UniVAD (CVPR 2025) ← target

## References

- Wang et al., [MedCLIP: Contrastive Learning from Unpaired Medical Images and Text](https://aclanthology.org/2022.emnlp-main.256/), EMNLP 2022.
- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), ICML 2021 (CLIP).
- Johnson et al., [MIMIC‑CXR: A Large Publicly Available Database of Labeled Chest Radiographs](https://arxiv.org/abs/1901.07042), 2019.
- Irvin et al., [CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison](https://arxiv.org/abs/1901.07031), AAAI 2019.
- Official MedCLIP repository: [github.com/RyanWangZf/MedCLIP](https://github.com/RyanWangZf/MedCLIP)

## License

MIT
