# XAI for Medical Text Classification
### Comparing LIME and Integrated Gradients on Medical Transcriptions

**Course:** Explainable AI (SOW-BKI266) — Radboud University 2025-2026

---

## Overview

This project applies two Explainable AI (XAI) methods to a transformer-based medical text classifier (DistilBERT) trained on clinical transcriptions. The goal is to compare **LIME** and **Integrated Gradients** in terms of faithfulness and interpretability in a medical NLP setting.

**Research Question:**
> How do LIME and Integrated Gradients compare in explaining the predictions of a transformer-based medical text classifier, in terms of faithfulness and interpretability?

---

## Repository Structure

```
xai-medical-text-classification/
├── xai_medical.ipynb                   # Main notebook (full pipeline)
├── README.md                           # This file
├── class_distribution.png              # Class distribution plot
├── training_loss.png                   # Training loss over epochs
├── lime_explanation.png                # LIME explanation output
├── integrated_gradients_explanation.png # IG explanation output
├── comparison_lime_vs_ig.png           # Side-by-side comparison
└── faithfulness_deletion_test.png      # Faithfulness evaluation plot
```

---

## Requirements

- Python 3.11
- VS Code with Jupyter extension (or Jupyter Notebook)

Install all dependencies by running the first cell in the notebook, or manually:

```bash
pip install transformers torch captum lime scikit-learn pandas numpy matplotlib seaborn
```

---

## Dataset

This project uses the **Medical Transcriptions** dataset from Kaggle:

1. Go to: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
2. Download `mtsamples.csv`
3. Place it in the **same folder** as `xai_medical.ipynb`

> Note: The dataset is not included in this repository due to Kaggle terms of use.

---

## How to Run

1. Clone this repository:
```bash
git clone https://github.com/[your-username]/xai-medical-text-classification.git
cd xai-medical-text-classification
```

2. Download `mtsamples.csv` from Kaggle and place it in the project folder.

3. Open `xai_medical.ipynb` in VS Code.

4. Run all cells in order from top to bottom.

The notebook will:
- Load and preprocess the dataset (top 5 medical specialties)
- Fine-tune DistilBERT for medical specialty classification
- Generate LIME explanations
- Generate Integrated Gradients explanations
- Produce a side-by-side comparison
- Run a faithfulness deletion test

---

## Results Summary

| Method | Type | Interpretability | Faithfulness |
|---|---|---|---|
| LIME | Model-agnostic, post-hoc | High (word-level) | Approximate |
| Integrated Gradients | Gradient-based, model-specific | Medium (subword tokens) | Axiomatic guarantees |

**Model accuracy:** 62% on test set (5-class classification, 3 epochs)

---

## XAI Methods Used

- **LIME** — `lime` library (`LimeTextExplainer`)
- **Integrated Gradients** — `captum` library (`LayerIntegratedGradients`)

---

## Reproducibility

All experiments use a fixed random seed (`SEED = 42`) for reproducibility.
