# 🧠 Improving PAE Classification with Data Augmentation in 3D CNNs

### How much augmentation is the right amount? A systematic sweep of augmentation rates for prenatal alcohol exposure classification from pediatric brain MRI

[![IEEE Xplore](https://img.shields.io/badge/IEEE%20Xplore-10783636-00629B?style=for-the-badge)](https://ieeexplore.ieee.org/document/10783636)
[![Venue](https://img.shields.io/badge/SIPAIM-2024%20·%20Antigua%2C%20Guatemala-B31B1B?style=for-the-badge)](https://www.sipaim.org/society/conferences)
[![Institution](https://img.shields.io/badge/University%20of-Calgary-D6001C?style=for-the-badge)](https://www.ucalgary.ca/)

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-augmentation-00A3E0?style=flat-square)
![Model](https://img.shields.io/badge/Model-3D%20CNN-FF6F00?style=flat-square)
![Modality](https://img.shields.io/badge/Modality-T1w%20structural%20MRI-D7263D?style=flat-square)
![Population](https://img.shields.io/badge/Population-pediatric-6A4C93?style=flat-square)
![Task](https://img.shields.io/badge/Task-binary%20classification-1D355C?style=flat-square)

[**⚡ Quick Start**](#-quick-start) · [**🔬 The Study**](#-the-study) · [**🔄 Augmentation**](#-augmentation) · [**📈 Training Curves**](#-training-curves) · [**📚 Cite**](#-citation)

---

## 📄 The Paper Behind This Repository

This repository contains the code and results for:

> **Improving Prenatal Alcohol Exposure Classification using Data Augmentation in 3D Convolutional Neural Networks**
> Kauê T. N. Duarte, Anik Das, Catherine Lebel, Mariana P. Bento
> *20th International Symposium on Medical Information Processing and Analysis (SIPAIM 2024)*, Antigua, Guatemala, 13–15 November 2024 · IEEE · pp. 1–4
> [`ieeexplore.ieee.org/document/10783636`](https://ieeexplore.ieee.org/document/10783636)

**The problem.** Prenatal alcohol exposure produces life-long consequences for learning, behaviour, and health, yet diagnosis remains hampered by inconsistent criteria and a shortage of objective biomarkers. Structural MRI is a promising substrate for automated detection — but the datasets are small. Pediatric neuroimaging cohorts number in the dozens to low hundreds, and every subject contributes exactly **one** training example to a 3D convolutional network with millions of parameters. That ratio is a recipe for overfitting.

**The idea.** Data augmentation is the standard answer to small-*n*, but "use augmentation" is not a specification — it is a dial, and nobody agrees on where to set it. Too little and the network memorises the training set; too much and it learns to be invariant to transformations that were carrying real anatomical signal, or simply drifts away from the distribution it will be tested on. This work treats the **augmentation rate itself as the object of study**, sweeping across rates and measuring the effect on both classification performance and generalisability rather than accepting a single default.

> [!NOTE]
> This work comes out of the **University of Calgary** — the Departments of Radiology, Engineering, and Biomedical Engineering — and builds on the group's earlier work applying transfer learning and explainability to PAE detection in pediatric MRI.

---

## 📑 Table of Contents

| | |
| --- | --- |
| **Background** | [The Study](#-the-study) · [Why Augmentation Rate Matters](#-augmentation) |
| **Usage** | [Quick Start](#-quick-start) · [Repository Contents](#-repository-contents) · [Requirements](#-requirements) |
| **Results** | [Training Curves](#-training-curves) |
| **Meta** | [Related Work](#-related-work) · [Citation](#-citation) · [Authors](#-authors--affiliations) · [License](#️-license) · [Contact](#️-contact) |

---

## ✨ Highlights

| **🎚️ Augmentation rate as the variable** Rather than fixing augmentation at a default and moving on, this study varies the rate systematically and reports what changes. | **🧊 3D throughout** Volumetric convolutions over T1-weighted structural MRI — no slice-wise approximation of a fundamentally 3D anatomy. |
| --- | --- |
| **📈 Full training curves published** `subplots.pdf` contains the curves for every model tested, not just the winner — so the overfitting behaviour is visible, not just summarised. | **🔁 Reproducible plotting** `generate_plot.py` regenerates every figure in the study from the recorded runs. |

---

## 🔬 The Study

Prenatal alcohol exposure classification from structural MRI is a **binary problem** — exposed versus unexposed — on volumetric T1-weighted scans of children. The architecture is a **3D convolutional neural network**; the experimental variable is how aggressively the training set is augmented.

The design is a sweep. Multiple models are trained under different augmentation rates, and each is evaluated on both:

- **Performance** — how well does it classify?
- **Generalisability** — does the gain hold on held-out data, or is it an artefact of a training set the model has effectively memorised?

Those two can move in opposite directions, which is exactly why the rate deserves study rather than a default value.

---

## 🔄 Augmentation

Augmentation for 3D medical volumes is not the same problem as augmentation for natural images. A horizontal flip is meaningful for a photograph of a cat and potentially destructive for a brain, where left–right asymmetry can itself be the signal. Transformations must stay inside the space of anatomically plausible variation.

`run_monai.py` implements the augmentation and training pipeline using [**MONAI**](https://monai.io/), the PyTorch-based framework built specifically for medical imaging — its transforms understand volumetric data and respect the spatial metadata that generic image libraries discard.

> [!TIP]
> When reporting augmentation experiments, state both the transform *and* its probability. "Random rotation" alone is underspecified — rotation applied to every scan with *p* = 1.0 is a categorically different intervention from rotation applied to a third of them, and the two will produce different generalisation behaviour on the same architecture.

> [!CAUTION]
> Augmentation must be applied to the **training split only**. Augmenting before splitting leaks transformed copies of the same subject across train and test, which inflates every metric and is invisible in the results table. With one scan per subject, this is the easiest mistake to make in this setup and the hardest to spot afterwards.

---

## ⚡ Quick Start

```bash
# 1 — Create and activate the environment
conda create -n pae python=3.9 -y
conda activate pae

# 2 — Install dependencies
pip install numpy pandas matplotlib scikit-learn monai torch

# 3 — Train / evaluate across augmentation rates
python run_monai.py

# 4 — Regenerate the figures
python generate_plot.py
```

---

## 📂 Repository Contents

```
PAE_Classification/
├── run_monai.py         # Augmentation + 3D CNN training pipeline (MONAI)
├── generate_plot.py     # Regenerates every figure used in the study
├── subplots.pdf         # Training curves for all models tested
└── README.md
```

| File | What it does |
| --- | --- |
| **`run_monai.py`** | The experimental pipeline — builds the 3D CNN, applies MONAI augmentation transforms at the configured rate, and runs training and evaluation. |
| **`generate_plot.py`** | Plotting code for all figures reported in the paper. |
| **`subplots.pdf`** | Training curves for every model configuration tested, collected into one document. |

---

## 📈 Training Curves

[**`subplots.pdf`**](subplots.pdf) collects the training curves for all tested models side by side. Reading them together is the point: a single accuracy number hides whether a model converged cleanly or oscillated its way to a lucky epoch, and the divergence between training and validation curves is where the effect of the augmentation rate actually becomes visible.

---

## 📦 Requirements

| Package | Purpose |
| --- | --- |
| **`monai`** | Medical-imaging transforms and 3D augmentation |
| **`torch`** | Backend for MONAI |
| **`numpy`** | Array handling |
| **`pandas`** | Results tabulation |
| **`scikit-learn`** | Metrics and cross-validation splits |
| **`matplotlib`** | Figure generation |

```bash
pip install numpy pandas matplotlib scikit-learn monai torch
```

> [!IMPORTANT]
> A **CUDA-capable GPU** is strongly recommended. 3D convolutions over full-resolution volumes are memory-hungry, and an augmentation sweep multiplies the number of training runs by the number of rates tested.

---

## 🔗 Related Work

This study sits alongside the group's other work on PAE and pediatric neuroimaging:

- Das A, Duarte KTN, Lebel C, Bento MP (2024). Deep learning for detecting prenatal alcohol exposure in pediatric brain MRI: a transfer learning approach with explainability insights. *Frontiers in Computational Neuroscience* 18:1434421. [`10.3389/fncom.2024.1434421`](https://doi.org/10.3389/fncom.2024.1434421)
- Das A, Duarte KTN, Wong S, Mamoun Y, Bento MP (2023). Comparative skull stripping techniques on pediatric magnetic resonance imaging. *19th International Symposium on Medical Information Processing and Analysis (SIPAIM)*, Mexico City, 1–4.

Other repositories from this author:

- [**fMRI_age_prediction**](https://github.com/KaueTND/fMRI_age_prediction) — structured ANNs for chronological brain age prediction from functional connectivity (SIPAIM 2025)
- [**Margarida_WMH_Seg_Toolbox**](https://github.com/KaueTND/Margarida_WMH_Seg_Toolbox) — multi-stage semi-supervised white matter hyperintensity segmentation (*Front. Comput. Neurosci.* 2024)

---

## 📚 Citation

If this code contributes to your work, please cite:

```bibtex
@inproceedings{duarte2024improving,
  title     = {Improving Prenatal Alcohol Exposure Classification using Data
               Augmentation in 3D Convolutional Neural Networks},
  author    = {Duarte, Kau{\^e} T. N. and Das, Anik and Lebel, Catherine and
               Bento, Mariana P.},
  booktitle = {2024 20th International Symposium on Medical Information
               Processing and Analysis (SIPAIM)},
  address   = {Antigua, Guatemala},
  publisher = {IEEE},
  pages     = {1--4},
  year      = {2024}
}
```

**Plain text**
> Duarte KTN, Das A, Lebel C, Bento MP (2024). Improving Prenatal Alcohol Exposure Classification using Data Augmentation in 3D Convolutional Neural Networks. *20th International Symposium on Medical Information Processing and Analysis (SIPAIM)*, Antigua, Guatemala, 1–4. IEEE.

---

## 👥 Authors & Affiliations

| Author | Affiliation | Contact |
| --- | --- | --- |
| **Kauê T. N. Duarte** | Department of Radiology, University of Calgary, AB, Canada | <kaue.duarte@ucalgary.ca> |
| **Anik Das** | Department of Engineering, University of Calgary, AB, Canada | <anik.das@ucalgary.ca> |
| **Catherine Lebel** | Department of Radiology, University of Calgary, AB, Canada | <clebel@ucalgary.ca> |
| **Mariana P. Bento** | Department of Biomedical Engineering, University of Calgary, AB, Canada | <mariana.pinheirobent@ucalgary.ca> |

---

## ⚖️ License

No license file is currently present in this repository. Without one, default copyright applies and others have no explicit permission to reuse the code — adding a `LICENSE` file (MIT, as used in this author's other repositories) would make the terms clear.

The associated conference paper is © 2024 IEEE. Reuse of the publication is governed by IEEE copyright policy.

> [!CAUTION]
> This is a **research tool**. It is not a certified medical device and must not be used for diagnosis or clinical decision-making. Prenatal alcohol exposure diagnosis is a clinical determination requiring multidisciplinary assessment.

---

## ✉️ Contact

Questions, bug reports, and feature requests are welcome via [**GitHub Issues**](https://github.com/KaueTND/PAE_Classification/issues).

**Kauê T. N. Duarte** — <kaue.duarte@ucalgary.ca>
Department of Radiology · University of Calgary

⭐ If this repository is useful to your work, a star helps others find it.
