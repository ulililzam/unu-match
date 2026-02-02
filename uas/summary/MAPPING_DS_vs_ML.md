# PEMETAAN KOMPONEN: DATA SCIENCE vs MACHINE LEARNING
## Project UNU-Match

---

## 📊 MATA KULIAH DATA SCIENCE

### Fokus: Data Processing, Analysis, & Insights

#### ✅ Komponen yang Termasuk:

| No | Komponen | File/Script | Penjelasan |
|----|----------|-------------|------------|
| 1 | **Data Collection** | `dataset_unu.csv` | Dataset 1001 records mahasiswa |
| 2 | **Data Generation** | `ml/generate_realistic_dataset.py` | Script generate synthetic data |
| 3 | **Data Cleaning** | `scripts/fix_dataset_v2.py`, `scripts/fix_dataset_kkm.py` | Pembersihan & validasi data |
| 4 | **Exploratory Data Analysis** | `scripts/analyze_dataset.py` | EDA, statistik deskriptif, distribusi |
| 5 | **Data Visualization** | Plots, charts, heatmaps | Visualisasi distribusi & korelasi |
| 6 | **Feature Engineering** | Normalization, scaling | Transform features ke [0,1] |
| 7 | **Data Splitting** | Train-test split 80-20 | Pembagian data untuk training & testing |
| 8 | **Statistical Analysis** | Correlation, variance | Analisis hubungan antar features |
| 9 | **Data Quality Check** | Missing values, outliers | Validasi kualitas dataset |
| 10 | **Target Encoding** | LabelEncoder | Encode prodi string → integer |

#### 📁 Files Data Science:

```
Data Science Components:
├── dataset_unu.csv                    # Dataset utama
├── ml/
│   └── generate_realistic_dataset.py  # Data generator
├── scripts/
│   ├── analyze_dataset.py             # EDA script
│   ├── fix_dataset_v2.py              # Data cleaning
│   └── fix_dataset_kkm.py             # KKM adjustment
└── uas/
    └── summary/
        └── DATA_SCIENCE_SUMMARY.md    # Summary DS
```

#### 🎯 Deliverables Data Science:

1. ✅ **Dataset bersih & tervalidasi** (1001 records, no missing values)
2. ✅ **Statistical analysis report** (distribusi, korelasi, outliers)
3. ✅ **Visualization outputs** (class distribution, correlation heatmap)
4. ✅ **Data splitting strategy** (stratified 80-20 split)
5. ✅ **Feature engineering pipeline** (normalization, encoding)
6. ✅ **Data quality report** (100% clean, balanced classes)

#### 📈 Key Insights dari Data Science:

- **Pattern Discovery:** Minat > Nilai (interest lebih prediktif)
- **Class Balance:** 8-14% per class (well distributed)
- **Feature Independence:** Low correlation (<0.35) antar features
- **Data Quality:** No missing values, no outliers ekstrem
- **STEM Pattern:** MTK & Fisika tinggi → Informatika/Teknik Elektro
- **Health Pattern:** Kimia & Biologi tinggi → Farmasi
- **Business Pattern:** Ekonomi tinggi + Minat Bisnis → Manajemen/Akuntansi

---

## 🤖 MATA KULIAH MACHINE LEARNING

### Fokus: Model Building, Training, & Prediction

#### ✅ Komponen yang Termasuk:

| No | Komponen | File/Script | Penjelasan |
|----|----------|-------------|------------|
| 1 | **Algorithm Selection** | Random Forest Classifier | Pilih algoritma supervised learning |
| 2 | **Model Architecture** | Ensemble of 300 trees | Design struktur model |
| 3 | **Hyperparameter Tuning** | GridSearchCV, manual tuning | Optimasi parameter model |
| 4 | **Model Training** | `ml/train_model_fast.py` | Training 300 decision trees |
| 5 | **Model Validation** | 5-fold cross-validation | Validasi performa model |
| 6 | **Model Evaluation** | Accuracy, F1, confusion matrix | Evaluasi multiple metrics |
| 7 | **Feature Importance** | Gini importance | Analisis kontribusi features |
| 8 | **Model Export** | JSON serialization | Export model untuk deployment |
| 9 | **Inference Engine** | `js/ml_engine.js` | Prediction system in browser |
| 10 | **Weighted Voting** | Enhanced RF voting | Custom voting dengan weights |
| 11 | **Business Rules** | `js/business_rules.js` | Hybrid ML + rules system |
| 12 | **Match Score** | Weighted factors | Multi-criteria scoring |
| 13 | **Model Testing** | `test_accuracy.html` | Test suite 6 scenarios |
| 14 | **Model Deployment** | Browser-based ML | Production deployment |

#### 📁 Files Machine Learning:

```
Machine Learning Components:
├── ml/
│   ├── train_model.py                # Full training pipeline
│   ├── train_model_fast.py           # Fast training (no grid search)
│   └── requirements.txt              # ML dependencies
├── models/
│   ├── rf_model.json                 # Trained model (JSON)
│   ├── rf_model.pkl                  # Python model backup
│   ├── feature_importance.json       # Feature weights
│   └── model_metadata.json           # Training metrics
├── js/
│   ├── ml_engine.js                  # Inference engine
│   ├── business_rules.js             # Validation rules
│   └── script.js                     # Prediction orchestration
├── test_accuracy.html                # Model testing
└── uas/
    └── summary/
        └── MACHINE_LEARNING_SUMMARY.md  # Summary ML
```

#### 🎯 Deliverables Machine Learning:

1. ✅ **Trained model** (Random Forest 300 trees)
2. ✅ **Performance metrics** (70% base, 86-90% effective)
3. ✅ **Feature importance analysis** (minat_teknik 13.38%)
4. ✅ **Model export** (JSON for browser deployment)
5. ✅ **Inference system** (JavaScript real-time prediction)
6. ✅ **Enhancement systems** (weighted voting, business rules)
7. ✅ **Test suite** (6 scenarios validation)
8. ✅ **Production deployment** (offline-ready web app)

#### 🎯 Performance Results:

- **Base Model Accuracy:** 70.0%
- **Cross-Validation:** 71.1% ± 4.3%
- **Training Time:** ~5 seconds
- **Inference Time:** 20-30 ms
- **Effective Accuracy:** 86-90% (with enhancements)

---

## 🔀 OVERLAP AREA (Digunakan Keduanya)

Beberapa komponen digunakan oleh kedua mata kuliah:

| Komponen | Data Science | Machine Learning | Keterangan |
|----------|-------------|------------------|------------|
| **Train-Test Split** | ✅ Splitting strategy | ✅ Model validation | DS: prepare data, ML: validate model |
| **Feature Names** | ✅ EDA analysis | ✅ Model input | Shared understanding |
| **Dataset** | ✅ Analysis object | ✅ Training data | Same source, different usage |
| **Normalization** | ✅ Preprocessing | ✅ Model input | DS: technique, ML: application |
| **Python Libraries** | ✅ Pandas, NumPy | ✅ Scikit-learn | Different focus |

---

## 📋 CHECKLIST SOAL UAS

### ✅ Requirements yang Sudah Dipenuhi:

#### 1. Mini Project Study Kasus ✅
- **Kasus:** Sistem Rekomendasi Jurusan Mahasiswa
- **Data:** Real-inspired dataset 1001 mahasiswa
- **Public/Real:** Synthetic realistic data based on real patterns

#### 2. Pre-Processing ✅
- **Data Science:** Data cleaning, validation, quality check
- **Scripts:** `fix_dataset_v2.py`, `analyze_dataset.py`
- **Result:** 100% clean data, no missing values

#### 3. Splitting Data ✅
- **Data Science:** Stratified 80-20 split
- **Machine Learning:** Training (800) vs Testing (200)
- **Validation:** 5-fold cross-validation

#### 4. Pemodelan Algoritma ✅
- **Machine Learning:** Random Forest Classifier
- **Type:** Supervised Learning (Classification)
- **Architecture:** Ensemble of 300 decision trees
- **Performance:** 70% base, 86-90% effective

#### 5. Supervised/Unsupervised ✅
- **Pilihan:** Supervised Learning
- **Alasan:** Labeled data (prodi diketahui), classification task
- **Algorithm:** Random Forest (ensemble method)

#### 6. Python Programming ✅
- **All scripts in Python:** ✅
  - `train_model_fast.py` (training)
  - `generate_realistic_dataset.py` (data generation)
  - `analyze_dataset.py` (EDA)
  - `fix_dataset_*.py` (preprocessing)

#### 7. Output Laporan ✅
- **Data Science Summary:** `uas/summary/DATA_SCIENCE_SUMMARY.md`
- **Machine Learning Summary:** `uas/summary/MACHINE_LEARNING_SUMMARY.md`
- **Mapping:** `uas/summary/MAPPING_DS_vs_ML.md` (this file)
- **Format:** Markdown (dapat di-convert ke PDF)

---

## 📤 FORMAT PENGUMPULAN

### Untuk Mata Kuliah Data Science:
```
Filename: [NAMATIM]_DataScience_UNU-Match_RecommendationSystem.pdf
Contents:
├── Cover (Nama tim, judul, mata kuliah)
├── Abstract
├── Pendahuluan (Problem statement)
├── Dataset (Collection, structure, characteristics)
├── Exploratory Data Analysis
│   ├── Statistical summary
│   ├── Correlation analysis
│   └── Visualization
├── Pre-Processing
│   ├── Data cleaning
│   ├── Feature engineering
│   └── Normalization
├── Data Splitting
│   ├── Train-test split strategy
│   └── Cross-validation setup
├── Insights & Findings
├── Conclusion
└── References

Source: uas/summary/DATA_SCIENCE_SUMMARY.md
```

### Untuk Mata Kuliah Machine Learning:
```
Filename: [NAMATIM]_MachineLearning_UNU-Match_RecommendationSystem.pdf
Contents:
├── Cover (Nama tim, judul, mata kuliah)
├── Abstract
├── Problem Formulation (Supervised learning, classification)
├── Algorithm Selection
│   ├── Why Random Forest
│   ├── Comparison with alternatives
│   └── Architecture design
├── Model Training
│   ├── Hyperparameter tuning
│   ├── Training process
│   └── Training results
├── Model Evaluation
│   ├── Performance metrics
│   ├── Classification report
│   ├── Confusion matrix
│   └── Cross-validation
├── Feature Importance Analysis
├── Model Deployment
│   ├── Export strategy
│   ├── Inference engine
│   └── Enhancement techniques
├── Results & Discussion
├── Conclusion
└── References

Source: uas/summary/MACHINE_LEARNING_SUMMARY.md
```

---

## 📊 PERBANDINGAN KONTRIBUSI

### Data Science (40-45%):
- Data collection & generation
- EDA & statistical analysis
- Data cleaning & preprocessing
- Feature engineering
- Data splitting strategy
- Visualization & insights

### Machine Learning (55-60%):
- Algorithm selection & justification
- Model architecture design
- Hyperparameter tuning
- Model training & validation
- Performance evaluation
- Feature importance analysis
- Model deployment & inference
- Enhancement systems

---

## 🎯 KESIMPULAN

### Data Science Component:
- ✅ Fokus: **Data → Insights**
- ✅ Output: Clean dataset, statistical analysis, patterns
- ✅ Tools: Pandas, NumPy, Matplotlib, Seaborn
- ✅ Deliverables: Dataset, EDA report, visualizations

### Machine Learning Component:
- ✅ Fokus: **Model → Predictions**
- ✅ Output: Trained model, performance metrics, deployment
- ✅ Tools: Scikit-learn, Random Forest, Model export
- ✅ Deliverables: Trained model, inference engine, test results

### Integration:
Data Science provides **foundation** → Machine Learning builds **intelligence**

**UNU-Match = Complete Data Science + Machine Learning Project** ✅

---

## 📞 PERTANYAAN UNTUK DOSEN/TIM

1. ✅ **Nama Tim:** Siapa saja anggota tim? (untuk cover laporan)
2. ✅ **Format Laporan:** 1 laporan unified atau 2 terpisah?
3. ✅ **Visualisasi:** Perlu tambah plots/charts dalam laporan?
4. ✅ **Presentasi:** Slide presentasi perlu dibuat?
5. ✅ **Demo:** Live demo aplikasi saat presentasi?

---

**Created by:** Mahasiswa Informatika  
**Date:** February 2, 2026  
**Purpose:** Mapping DS vs ML components untuk 2 mata kuliah  
**Project:** UNU-Match v1.0
