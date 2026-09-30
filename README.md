# Myopia Risk Prediction: Machine Learning Benchmark & Interpretability Framework

An end-to-end, leakage-controlled machine learning pipeline that replicates and extends the benchmark study by **Mu et al. (2024)**[cite: 2]. This project predicts childhood myopia risk using lifestyle, dietary, and ocular biometric data across **10 machine learning models**, featuring rigorous statistical testing and multi-method explainability (SHAP, Permutation Importance, Odds Ratios).

> **Note:** Built on a hybrid research cohort ($N = 1,500$). Results demonstrate pipeline methodology and reproducibility rather than actionable clinical evidence.

---

## 📌 Project Overview

- **Problem:** Predict whether a child is myopic ($\text{SE} \le -0.50\text{ D}$) using non-invasive clinical and lifestyle indicators.
- **Replication:** Benchmarked the 5 baseline models from Mu et al. (Logistic Regression, Decision Tree, Random Forest, XGBoost, Support Vector Machine).
- **Extensions:** Added 5 modern tabular models—LightGBM, a 4-model Stacking Ensemble, an Entity-Embedding MLP, an FT-Transformer, and an attention-free FT-Transformer ablation[cite: 2, 4].
- **Key Takeaway:** Simple Logistic Regression matched or outperformed complex deep architectures (Holdout AUC: **0.983**, 95% CI: 0.973–0.991; 5×10 CV AUC: **0.978 ± 0.006**)[cite: 4]. Stripping self-attention from the FT-Transformer resulted in zero loss of performance, proving self-attention offers no distinct advantage on this tabular scale[cite: 4].

---

## 🚀 Key Features & Methodological Rigor

- **Strict Leakage Prevention:** Excluded Spherical Equivalent (SE) and derived severity labels from inputs[cite: 2, 4]. All preprocessing, scaling, and feature selection (LASSO) are refit strictly inside cross-validation folds[cite: 2, 4].
- **Reproducible Architecture:** Modular 6-notebook Colab pipeline linked via Google Drive with SHA-256 hash manifest verification between steps (`artifacts/manifest.json`)[cite: 2, 3, 4]. Fixed seed (`42`) across Python, NumPy, and PyTorch[cite: 2, 3, 4].
- **Statistical Rigor:** 5×10 Repeated Stratified Cross-Validation (50 total folds), 2,000-draw bootstrap 95% Confidence Intervals, and pairwise DeLong tests with Holm-Bonferroni correction across all 45 model pairs[cite: 2, 4].
- **Triangulated Explainability:** Converged risk factor attributions using SHAP (Beeswarm, Waterfall), Permutation Importance, and Multivariable Logistic Regression Odds Ratios[cite: 2, 4].

---

## 📊 Summary of Model Performance

Evaluated on a 70/30 stratified train-test split ($N_{\text{train}} = 1,050$, $N_{\text{test}} = 450$)[cite: 4]:

| Model | Group | Holdout Acc | Holdout F1 | Holdout AUC (95% CI) | 5×10 CV AUC (Mean ± SD) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Baseline | **0.929** | **0.892** | **0.983 (0.973–0.991)** | **0.978 ± 0.006** 🏆 |
| **Stacking Ensemble** | Extension | 0.938 | 0.903 | 0.981 (0.968–0.990) | 0.977 ± 0.006 |
| **XGBoost** | Baseline | 0.924 | 0.886 | 0.980 (0.965–0.990) | 0.975 ± 0.007 |
| **FT-Transformer** | Extension | 0.904 | 0.863 | 0.978 (0.963–0.989) | 0.972 ± 0.009 |
| **LightGBM** | Extension | 0.920 | 0.878 | 0.977 (0.962–0.988) | 0.973 ± 0.008 |
| **Random Forest** | Baseline | 0.933 | 0.894 | 0.976 (0.962–0.987) | 0.972 ± 0.008 |
| **SVM (RBF)** | Baseline | 0.927 | 0.885 | 0.974 (0.959–0.985) | 0.972 ± 0.006 |
| **FT-Trans. (No Attn)**| Ablation | 0.922 | 0.883 | 0.973 (0.958–0.984) | 0.976 ± 0.006 |
| **Decision Tree** | Baseline | 0.920 | 0.876 | 0.968 (0.949–0.983) | 0.960 ± 0.010 |
| **MLP (PyTorch)** | Extension | 0.904 | 0.860 | 0.967 (0.952–0.980) | 0.977 ± 0.006 |

---

## 💡 Key Clinical & XAI Insights

Across SHAP, Permutation Importance, and Multivariable Odds Ratios, three factors emerged as primary risk drivers[cite: 4]:
1. **Axial Length (`axial_length_mm`):** Primary biometric driver ($\text{OR} = 215.7$ per SD, $p < 0.001$)[cite: 4].
2. **Family History (`parental_myopia_count`):** Strong genetic indicator ($\text{OR} = 3.36$ per parent, $p < 0.001$)[cite: 4].
3. **Dietary Sugar (`diet_sugary_freq`):** Significant lifestyle predictor ($\text{OR} = 2.75$, $p = 0.001$)[cite: 4].
4. **Outdoor Activity (`outdoor_hours_daily`):** Key protective factor ($\text{OR} = 0.60$ per SD, $p = 0.011$)[cite: 4].

---

## 🛠️ Pipeline Architecture

```text
Notebooks/
├── SoftCom_01_Data_Validation.ipynb   # Schema checks, value domains & cohort statistics
├── SoftCom_02_Preprocessing.ipynb     # Feature engineering, ratio creation & 70/30 split
├── SoftCom_03_Benchmarking.ipynb      # GridSearchCV, 5x10 CV, model training & DeLong tests
├── SoftCom_04_Explainable_AI.ipynb    # Permutation importance, Kernel/TreeSHAP & Odds Ratios
├── SoftCom_05_Figures.ipynb           # Publication-ready charts (ROC, Heatmap, Prevalence)
└── SoftCom_06_Export.ipynb            # Integrity checks & automated JSON summary compilation
