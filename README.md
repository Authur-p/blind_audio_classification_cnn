# 🎧 Blind Audio Classification — MFCC vs STFT with CNNs

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![Librosa](https://img.shields.io/badge/Librosa-4B0082?style=flat-square)

Classifying **sustained-vowel audio signals** into two classes using convolutional neural networks, and comparing feature-extraction strategies — **MFCC** vs **STFT spectrograms** — on a *blind* dataset (class meanings unknown). Built for the **EE992** group project (MSc, University of Strathclyde).

> **TL;DR** — A controlled feature-extraction study on ~1,610 audio clips. MFCC-CNN gave the most consistent macro-F1 (**0.68 ± 0.019** across 5 folds), beating a ~0.50 majority-class baseline, with better stability and lower compute than STFT.

---

## 🎯 Problem

- **Data:** ~1,610 raw `.wav` clips of sustained vowels, binary labels (class 0: 543, class 1: 1,067 — a ~1:2 imbalance). Blind challenge — the classes are unlabelled, so the emphasis is on the model's ability to separate them.
- **Context:** sustained-vowel analysis is used clinically (e.g. COPD and voice-disorder screening).

## 🔬 Approach

**Feature extraction (compared head-to-head)**
- **MFCC:** 40 coefficients + first/second-order deltas → 120 × 94 feature maps.
- **STFT spectrograms** classified with a **MobileNet** transfer-learning backbone (ImageNet weights).
- **Mel-spectrograms** as a third comparison point.

**Pre-processing**
- Fixed frame width of 94 (~3 s at 16 kHz, hop 512) chosen empirically over 63 for a more stable decision threshold.
- Volume normalisation; length standardised to the 25th-percentile duration to avoid learning padding.
- **Augmentation** (noise, time-stretch, pitch-shift) tripled the training set to 3,864; `class_weight='balanced'` to counter imbalance.

**Model &amp; evaluation**
- CNN: three 3×3 conv blocks (batch-norm, 2×2 max-pool, increasing filters) + global average pooling + dropout.
- **5-fold stratified cross-validation**, early stopping on `val_loss`.
- **Macro-F1** as the primary metric; per-fold decision threshold tuned (median 0.59) rather than a fixed 0.5, given the class imbalance.

## 📊 Results

| Feature extraction | Input shape | Mean accuracy | Mean macro-F1 | F1 std |
| :-- | :-- | :-- | :-- | :-- |
| **MFCC** *(selected)* | 120 × 94 | 0.710 | **0.683** | **0.019** |
| Mel-spectrogram | 128 × 392 | 0.726 | 0.676 | 0.040 |
| STFT + MobileNet | 640 × 480 | 0.714 | 0.689 | 0.038 |

**MFCC won on consistency and efficiency** — comparable F1 to STFT with roughly half the variance and far less compute. Macro-F1 of 0.68 is a clear lift over the ~0.50 a majority-class predictor would score, while honestly reflecting a genuinely hard, small, imbalanced dataset.

## 🗂️ Repository

```
├── ee992_groupf_project.ipynb     # feature extraction, CNN, CV, evaluation
├── EE992_Report_GroupF.pdf        # IEEE-style written report
└── README.md
```

## ▶️ Run it

```bash
pip install tensorflow librosa scikit-learn numpy matplotlib
jupyter notebook ee992_groupf_project.ipynb
```

## 👥 Team

Group F, EE992 — **Daniel Opeyemi** (MSc, EEE) and James Slavin (PhD candidate, EEE), University of Strathclyde.

## 👤 Author of this repo

**Daniel Opeyemi** — [Portfolio](https://authur-p.github.io/) · [LinkedIn](https://www.linkedin.com/in/daniel-opeyemi-i/)
