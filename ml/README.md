# 🤖 Machine Learning Implementation Guide

## Overview

UNU-Match menggunakan **Random Forest Classifier** yang di-train dengan Python scikit-learn dan di-deploy ke JavaScript.

**Performance:**
- Test Accuracy: **91.0%**
- Cross-Validation F1: **93.98% (±1.74%)**
- Algorithm: Random Forest (200 trees, max_depth=15)

---

## 🔍 Feature Importance

```
1. Bahasa Inggris      12.97%
2. Ekonomi             10.97%
3. Biologi             10.43%
4. Minat Pendidikan    10.39%
5. Fisika               9.24%
6. Agama                9.08%
7. Minat Kesehatan      8.44%
8. Kimia                8.35%
9. Matematika           7.81%
10. Minat Teknik        7.14%
11. Minat Bisnis        4.15%
12. Hafalan             1.02%
```

---

## 📂 Project Structure

```
unu-match/
├── ml/
│   ├── train_model.py          # Main training script
│   └── requirements.txt
├── models/
│   ├── rf_model.json           # Exported model for JS
│   ├── feature_importance.json
│   └── model_metadata.json
├── js/
│   ├── script.js
│   └── ml_engine.js            # ML inference engine
└── dataset_unu.csv             # Training data (1000 records)
```

---

## 🚀 How to Retrain Model

### Setup & Train:

```bash
cd ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py
```

### Optional - Tune Hyperparameters:

Edit `train_model.py`:
```python
trainer.run_full_pipeline(tune_hyperparameters=True)
```

---

## 🔧 Model Configuration

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)
```

---

## ⚠️ Known Limitations

1. **Akuntansi vs Manajemen** - Kedua jurusan punya profil mirip, performa lebih rendah
2. **Model Size** - rf_model.json ~2-3 MB (50 dari 200 trees untuk efisiensi)

---

**Last Updated:** January 20, 2026  
**Model Version:** 1.0  
**Accuracy:** 91.0%
