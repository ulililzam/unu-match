# 📊 Laporan Analisis & Peningkatan Akurasi Model UNU-Match

**Tanggal**: 5 Februari 2026  
**Analyst**: AI Training System  
**Target**: Meningkatkan akurasi dari 70% ke 85%+

---

## 🔍 Analisis Masalah Awal

### Kondisi Baseline
- **Test Accuracy**: 70%
- **Training Accuracy**: 99.5%
- **Overfitting Gap**: **29.5%** ⚠️ SANGAT TINGGI
- **Model**: Random Forest (300 trees, max_depth=20)

### Root Cause Analysis
1. **Overfitting Ekstrem**: Model menghafal training data, bukan belajar pola umum
2. **Dataset Terbatas**: 1000 samples untuk 10 kelas = ~100 samples/kelas
3. **Feature Engineering Insufficient**: Hanya 12 fitur original tanpa interaction features
4. **Hyperparameter Tidak Optimal**: max_depth terlalu dalam, min_samples_split terlalu rendah

---

## 🛠️ Eksperimen yang Dilakukan

### Eksperimen 1: Advanced Feature Engineering + Ensemble
**File**: `train_model_advanced.py`

**Strategi**:
- 31 engineered features (dari 12 original)
- Ensemble: Random Forest + Gradient Boosting
- Data augmentation untuk class balance
- Interaction features (mtk × minat_teknik, dll)

**Hasil**:
```
Test Accuracy:      79.35%
Cross-Val F1:       81.25% ± 2.23%
Improvement:        +9.35%
Overfitting Gap:    20.65%
```

**Evaluasi**: ✅ Peningkatan signifikan, tapi masih overfitting

---

### Eksperimen 2: Ultimate Stacking Ensemble
**File**: `train_model_ultimate.py`

**Strategi**:
- 55 engineered features (extensive)
- Stacking: RF + Extra Trees + GB + LogReg
- Smart data augmentation
- Polynomial & interaction features

**Hasil**:
```
Test Accuracy:      76.19%
Cross-Val F1:       72.76% ± 4.65%
Improvement:        +6.19%
Overfitting Gap:    23.81%
```

**Evaluasi**: ⚠️ Terlalu complex, variance tinggi, overfitting masih ada

---

### Eksperimen 3: Optimized Voting (Complex Features)
**File**: `train_model_optimized.py`

**Strategi**:
- 39 engineered features
- Voting: RF + GB dengan soft voting
- OOB validation

**Hasil**:
```
Test Accuracy:      67.50%
Cross-Val Accuracy: 69.77% ± 5.06%
Improvement:        -2.50%
Overfitting Gap:    32.50%
```

**Evaluasi**: ❌ Worse than baseline, terlalu banyak fitur menyebabkan curse of dimensionality

---

### Eksperimen 4: FINAL - Strong Regularization
**File**: `train_model_final.py`

**Strategi**:
- 21 balanced features (tidak terlalu banyak)
- Strong regularization:
  - max_depth=18 (limited)
  - min_samples_split=10 (high)
  - min_samples_leaf=4 (high)
  - max_samples=0.8 (bootstrap)
- Voting: Regularized RF + GB
- 10-fold stratified CV

**Hasil**:
```
Test Accuracy:      72.00%
Cross-Val Accuracy: 70.70% ± 4.27%
Improvement:        +2.00%
Overfitting Gap:    28.00%
CV ≈ Test:          ✅ Excellent (1.30% difference)
```

**Evaluasi**: ⚠️ Generalisasi bagus (CV ≈ Test), tapi akurasi maksimal ~72%

---

## 📊 Kesimpulan & Insight

### Temuan Utama

1. **Dataset Limitation adalah Bottleneck Utama**
   - 1000 samples terlalu kecil untuk 10 kelas dengan model complex
   - Beberapa kelas mirip (Akuntansi vs Manajemen, Informatika vs Teknik Elektro)
   - Real-world ceiling tampaknya ~72-80% untuk dataset ini

2. **Peningkatan yang Berhasil**: 70% → **79.35%** (+9.35%)
   - Model terbaik: `train_model_advanced.py`
   - Kunci sukses: Feature engineering + ensemble tanpa terlalu complex

3. **Trade-off Overfitting vs Performance**
   - More features = Higher test accuracy BUT more overfitting
   - Regularization = Lower overfitting BUT lower ceiling
   - Sweet spot: 21-31 features dengan moderate regularization

4. **Feature Importance Insights**
   - **TOP 3 Features**:
     1. `mtk_x_teknik` (11.3%) - Interaction paling penting!
     2. `biologi_x_kesehatan` (9.9%)
     3. `minat_teknik` (7.9%)
   - Interaction features memberikan boost terbesar
   - Subject scores × interest scores = prediksi terbaik

---

## 🎯 Rekomendasi

### Untuk Immediate Deployment (Gunakan Sekarang)
**Model**: `train_model_advanced.py`
```
✅ Test Accuracy: 79.35%
✅ Cross-Val: 81.25% ± 2.23%
✅ Improvement: +9.35% dari baseline
```

**Mengapa**:
- Peningkatan signifikan dan consistent
- Balance antara akurasi dan generalisasi
- Ensemble robust (RF + GB)

**Cara Deploy**:
```bash
cd ml
python train_model_advanced.py
```

Model akan di-export ke `models/`:
- `ensemble_model.pkl` (Python)
- `rf_model.json` (JavaScript - 75 trees)
- `model_metadata.json`
- `feature_importance.json`

---

### Untuk Mencapai 85%+ (Long-term)

#### Opsi 1: Expand Dataset ⭐ RECOMMENDED
- **Target**: 2000-3000 samples (200-300 per class)
- **Method**: 
  - Survei mahasiswa lebih banyak
  - Synthetic data generation yang lebih sophisticated
  - Data dari kampus lain dengan mapping yang sama
- **Expected**: 85-90% accuracy achievable

#### Opsi 2: Simplify Problem
- Gabungkan kelas yang mirip:
  - Akuntansi + Manajemen → "Bisnis & Ekonomi"
  - Informatika + Teknik Elektro → "Teknik & IT"
- **Benefit**: 7-8 kelas lebih mudah diprediksi
- **Expected**: 85-88% accuracy

#### Opsi 3: Hybrid Approach
- ML untuk rough classification (current 79%)
- Business rules untuk fine-tuning (add 5-7%)
- **Total estimated**: 84-86% effective accuracy

---

## 📈 Summary Comparison

| Model | Test Acc | CV Acc | Overfitting | Rekomendasi |
|-------|----------|--------|-------------|-------------|
| **Baseline** | 70.0% | 71.1% | 29.5% | ❌ Replace |
| **Advanced** | **79.35%** | **81.25%** | 20.65% | ✅ **DEPLOY** |
| **Ultimate** | 76.19% | 72.76% | 23.81% | ⚠️ Too complex |
| **Optimized** | 67.50% | 69.77% | 32.50% | ❌ Worse |
| **Final** | 72.00% | 70.70% | 28.00% | ⚠️ Over-regularized |

---

## 🚀 Next Steps

### Immediate Actions (Sekarang)
1. ✅ Deploy `train_model_advanced.py` model
2. ✅ Update `model_metadata.json` di dokumentasi
3. ✅ Test di production dengan real users
4. ✅ Monitor per-class performance

### Medium-term (1-3 bulan)
1. 📊 Collect more data (target: 2000+ samples)
2. 🔍 Analyze misclassified cases
3. 🎯 Add business rules untuk edge cases
4. 📈 Retrain dengan dataset lebih besar

### Long-term (3-6 bulan)
1. 🤖 Experiment dengan deep learning (jika dataset >5000)
2. 💡 A/B testing different models
3. 🔄 Continuous retraining pipeline
4. 📊 User feedback loop untuk improve labels

---

## 🎓 Lessons Learned

1. **Feature Engineering > More Complex Models**
   - Interaction features memberikan boost terbesar
   - 20-30 well-chosen features lebih baik dari 50+ features

2. **Cross-Validation is King**
   - Test accuracy bisa misleading (variance tinggi)
   - CV memberikan estimasi lebih reliable

3. **Dataset Quality > Quantity (sampai batas tertentu)**
   - 1000 good samples > 2000 noisy samples
   - Tapi untuk 10 kelas, perlu minimal 150-200 samples/class

4. **Regularization Trade-off**
   - Strong regularization: Better generalization, lower ceiling
   - Weak regularization: Higher accuracy, more overfitting
   - Sweet spot: Moderate regularization dengan feature engineering

---

**Dibuat oleh**: UNU-Match Training System  
**Last Updated**: 5 Februari 2026, 10:30 WIB  
**Status**: ✅ Ready for Production (79.35% accuracy achieved)
