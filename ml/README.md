# 🤖 Machine Learning Implementation Guide

## Overview

UNU-Match sekarang menggunakan **Random Forest Classifier** yang di-train dengan Python scikit-learn dan di-deploy ke JavaScript untuk inference yang cepat dan akurat.

---

## 🎯 **HASIL PENINGKATAN**

### **Before (K-NN):**
- Algorithm: K-Nearest Neighbors (K=150)
- Accuracy: **~75%** (estimated, no validation)
- Issues:
  - ❌ No scientific validation
  - ❌ Arbitrary K selection
  - ❌ No feature importance
  - ❌ Equal weights for all features
  - ❌ Sensitive to noise

### **After (Random Forest):**
- Algorithm: Random Forest (200 trees, max_depth=15)
- Test Accuracy: **91.0%**
- Cross-Validation F1: **93.98% (±1.74%)**
- Benefits:
  - ✅ Scientifically validated with train/test split
  - ✅ Feature importance analysis
  - ✅ Confidence scores
  - ✅ Robust to outliers
  - ✅ Explainable predictions

---

## 📊 **Model Performance Metrics**

```
Training Accuracy:   100.00%
Test Accuracy:       91.00%
Training F1:         1.0000
Test F1:             0.9088
Cross-Val F1:        0.9398 (+/- 0.0174)
```

### **Per-Class Performance:**

| Program Studi | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| S1 Agribisnis | 1.000 | 1.000 | 1.000 |
| S1 Akuntansi | 0.478 | 0.647 | 0.550 |
| S1 Farmasi | 1.000 | 1.000 | 1.000 |
| S1 Informatika | 1.000 | 1.000 | 1.000 |
| S1 Manajemen | 0.571 | 0.400 | 0.471 |
| S1 PGSD | 1.000 | 1.000 | 1.000 |
| S1 Pendidikan Bahasa Inggris | 1.000 | 1.000 | 1.000 |
| S1 Studi Islam Interdisipliner | 1.000 | 1.000 | 1.000 |
| S1 Teknik Elektro | 1.000 | 1.000 | 1.000 |
| S1 Teknologi Hasil Pertanian | 1.000 | 1.000 | 1.000 |

**Note:** Akuntansi dan Manajemen memiliki performa lebih rendah karena overlap fitur dengan jurusan bisnis lainnya.

---

## 🔍 **Feature Importance**

Berikut urutan fitur berdasarkan kontribusi terhadap prediksi:

```
1. Bahasa Inggris      12.97%  ██████
2. Ekonomi             10.97%  █████
3. Biologi             10.43%  █████
4. Minat Pendidikan    10.39%  █████
5. Fisika               9.24%  ████
6. Agama                9.08%  ████
7. Minat Kesehatan      8.44%  ████
8. Kimia                8.35%  ████
9. Matematika           7.81%  ███
10. Minat Teknik        7.14%  ███
11. Minat Bisnis        4.15%  ██
12. Hafalan             1.02%   
```

**Insight:**
- **Bahasa Inggris** paling penting karena hampir semua jurusan membutuhkan
- **Ekonomi & Biologi** memisahkan jurusan bisnis dan sains
- **Hafalan** penting hanya untuk Studi Islam Interdisipliner

---

## 🏗️ **Architecture**

### **Hybrid ML Pipeline:**

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: TRAINING (Python - Offline)                       │
├─────────────────────────────────────────────────────────────┤
│ 1. Load dataset_unu.csv (1000 records)                     │
│ 2. Feature engineering & normalization                      │
│ 3. Train/Test split (80/20, stratified)                    │
│ 4. Train Random Forest (200 trees)                         │
│ 5. Cross-validation (5-fold)                               │
│ 6. Feature importance analysis                              │
│ 7. Export model to JSON                                     │
│ 8. Save metadata & metrics                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: DEPLOYMENT (JavaScript - Online)                  │
├─────────────────────────────────────────────────────────────┤
│ 1. Load rf_model.json in browser                           │
│ 2. User fills survey form                                   │
│ 3. ML Engine predicts with Random Forest                   │
│ 4. Calculate probabilities & confidence                     │
│ 5. Explain predictions (feature contributions)             │
│ 6. Display top 3 recommendations                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 **Project Structure**

```
unu-match/
├── ml/                          # ML Training Scripts
│   ├── train_model.py          # Main training script
│   ├── requirements.txt        # Python dependencies
│   └── README.md               # This file
│
├── models/                      # Exported Models
│   ├── rf_model.json           # Random Forest (50 trees)
│   ├── feature_importance.json # Feature rankings
│   ├── model_metadata.json     # Metrics & parameters
│   ├── rf_model.pkl            # Python model (for retraining)
│   └── label_encoder.pkl       # Class encoder
│
├── js/
│   ├── script.js               # Main application logic
│   └── ml_engine.js            # ML inference engine
│
└── dataset_unu.csv             # Training data (1000 records)
```

---

## 🚀 **How to Retrain Model**

### **1. Setup Python Environment:**

```bash
cd ml
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### **2. Train Model:**

```bash
python train_model.py
```

**Output:**
- Model performance metrics
- Feature importance analysis
- Exported files in `models/` directory

### **3. (Optional) Tune Hyperparameters:**

Edit `train_model.py` line 340:

```python
trainer.run_full_pipeline(tune_hyperparameters=True)  # Enable GridSearch
```

**Warning:** Hyperparameter tuning takes 10-30 minutes!

---

## 🔧 **Configuration**

### **Model Parameters:**

```python
RandomForestClassifier(
    n_estimators=200,      # Number of trees
    max_depth=15,          # Max tree depth
    min_samples_split=5,   # Min samples to split
    min_samples_leaf=2,    # Min samples at leaf
    max_features='sqrt',   # Features per split
    random_state=42        # Reproducibility
)
```

### **Training Configuration:**

```python
test_size=0.2              # 20% for testing
cv=5                       # 5-fold cross-validation
stratify=True              # Maintain class balance
```

---

## 📈 **How Prediction Works**

### **Step-by-Step:**

1. **User Input:**
   ```javascript
   {
       mtk: 90,
       inggris: 85,
       agama: 70,
       fisika: 80,
       kimia: 65,
       biologi: 60,
       ekonomi: 70,
       minat_teknik: 5,
       minat_kesehatan: 2,
       minat_bisnis: 3,
       minat_pendidikan: 2,
       hafalan: 0
   }
   ```

2. **Feature Normalization:**
   ```javascript
   // Grades: 0-100 → 0-1
   mtk_normalized = 90 / 100 = 0.9
   
   // Interest: 1-5 → 0-1
   minat_teknik_normalized = (5 - 1) / 4 = 1.0
   ```

3. **Random Forest Prediction:**
   - Each of 50 trees votes for a class
   - Tree votes are aggregated
   - Probabilities calculated from vote counts

4. **Result:**
   ```javascript
   {
       topPredictions: [
           {
               rank: 1,
               prodi: "S1 Informatika",
               probability: 0.68,
               matchPercentage: 68.0,
               confidence: "Tinggi"
           },
           {
               rank: 2,
               prodi: "S1 Teknik Elektro",
               probability: 0.18,
               matchPercentage: 18.0,
               confidence: "Sedang"
           },
           {
               rank: 3,
               prodi: "S1 Akuntansi",
               probability: 0.08,
               matchPercentage: 8.0,
               confidence: "Rendah"
           }
       ]
   }
   ```

---

## 🎓 **Explainability**

### **Feature Contribution Example:**

For "S1 Informatika" prediction:

```
Top Contributing Features:
1. Matematika: 90 (kontribusi 11.8%)
2. Bahasa Inggris: 85 (kontribusi 11.0%)
3. Minat Teknik: 5/5 (kontribusi 10.2%)
4. Fisika: 80 (kontribusi 7.4%)
5. Agama: 70 (kontribusi 6.4%)
```

**Explanation:**
"Karena nilai Matematika tinggi (90), Bahasa Inggris kuat (85), dan minat teknik sangat tinggi (5/5), profil Anda sangat cocok dengan Informatika."

---

## ⚠️ **Known Limitations**

1. **Akuntansi vs Manajemen Confusion:**
   - Kedua jurusan punya profil mirip (ekonomi, bisnis)
   - Solution: Tambah fitur pembeda (minat accounting vs leadership)

2. **Model Size:**
   - rf_model.json = ~2-3 MB (50 trees)
   - Exported only 50 dari 200 trees untuk reduce file size
   - Trade-off: slight accuracy drop (~1-2%)

3. **Browser Compatibility:**
   - Requires ES6+ JavaScript
   - JSON loading via fetch() (needs server or file:// protocol)

---

## 🔮 **Future Improvements**

### **Short Term:**
- [ ] Add confidence intervals
- [ ] Implement ensemble methods (RF + Rule-based)
- [ ] A/B testing with real users

### **Medium Term:**
- [ ] Collect more training data (target: 5000+ samples)
- [ ] Add personality traits (MBTI, Big Five)
- [ ] Multi-label prediction (primary + backup jurusan)

### **Long Term:**
- [ ] Deep Learning (Neural Network)
- [ ] Real-time model updates
- [ ] Personalized recommendations based on user feedback

---

## 📞 **Support**

Jika ada pertanyaan atau issue terkait ML implementation:

1. Check console logs di browser (F12)
2. Verify model files di `/models/` directory
3. Re-run training: `python ml/train_model.py`
4. Report bugs di GitHub Issues

---

## 📚 **References**

- **scikit-learn Documentation:** https://scikit-learn.org/
- **Random Forest Paper:** Breiman, L. (2001). "Random Forests"
- **Feature Importance:** https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

---

**Last Updated:** January 18, 2026  
**Model Version:** 1.0  
**Accuracy:** 91.0%
