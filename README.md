# UNU-Match - Sistem Rekomendasi Jurusan Mahasiswa

Sistem cerdas berbasis web yang membantu calon mahasiswa Universitas Nahdlatul Ulama menemukan program studi yang paling cocok berdasarkan nilai akademik, minat, dan preferensi belajar.

---

## Tentang Project

**UNU-Match** adalah sistem rekomendasi jurusan yang dirancang khusus untuk membantu calon mahasiswa memilih program studi yang tepat. Sistem ini menggunakan algoritma **Random Forest Machine Learning** yang telah dilatih dengan dataset 1000+ data mahasiswa untuk memberikan rekomendasi yang akurat dan terpercaya.

### Mengapa UNU-Match?

- **User-Friendly**: Interface sederhana tanpa istilah teknis
- **Akurat**: Berbasis data real dari 1001 mahasiswa
- **Cepat**: Hanya 5 menit untuk mendapatkan hasil
- **Mobile-Responsive**: Bisa diakses dari HP, tablet, atau desktop
- **Offline-Ready**: Semua proses dilakukan di browser, tidak perlu internet setelah halaman dimuat

---

## Fitur Utama

### 1. Survey Interaktif
- **Slider Nilai Raport**: Input nilai 7 mata pelajaran dengan slider intuitif
- **Star Rating Minat**: Rating 1-5 bintang untuk 4 bidang minat
- **Toggle Hafalan**: Switch sederhana untuk preferensi hafalan
- **Progress Indicator**: Visual feedback progress pengisian
- **Auto-Save**: Jawaban tersimpan otomatis di browser

### 2. Random Forest Machine Learning
- **Algorithm**: Random Forest Classifier dengan 200 decision trees
- **Accuracy**: 73.5% test accuracy (realistic untuk education ML)
- **Feature Importance**: Interest-driven (minat_teknik 14%, minat_bisnis 10%)
- **Top 3 Recommendations**: Memberikan 3 rekomendasi terbaik dengan confidence level

### 3. Hasil Comprehensive
- **Match Percentage**: Tingkat kecocokan dalam persentase
- **Program Details**: Deskripsi lengkap program studi
- **Mata Kuliah**: Mata pelajaran utama yang dipelajari
- **Prospek Karir**: 4-5 pilihan karir untuk setiap jurusan
- **Visual Bars**: Visualisasi match percentage dengan animasi

### 4. Additional Features
- **Print Results**: Cetak hasil untuk dokumentasi
- **Share Results**: Bagikan hasil via Web Share API atau copy to clipboard
- **Restart Survey**: Mulai ulang dengan konfirmasi
- **Responsive Design**: Optimal di semua ukuran layar

---

## Teknologi

- **HTML5, CSS3, JavaScript (ES6+)**: Pure web technologies, no frameworks
- **TailwindCSS**: Utility-first CSS framework
- **Hugeicons**: Professional icon library
- **Google Fonts (Inter)**: Modern typography
- **LocalStorage API**: Client-side data persistence
- **Random Forest ML**: Trained with scikit-learn

---

## Cara Menggunakan

### Untuk End User

1. **Buka Aplikasi**
   - Double-click `index.html` atau
   - Jalankan local server (lihat Quick Start)

2. **Mulai Survey**
   - Klik tombol "Mulai Sekarang"
   - Jawab 3 langkah pertanyaan

3. **Lihat Hasil**
   - Klik "Lihat Hasil"
   - Dapatkan 3 rekomendasi jurusan
   - Cetak atau bagikan hasilnya

### Quick Start (Developers)

```bash
# Clone atau download project
cd unumatch

# Option 1: Python Simple Server
python3 -m http.server 8000

# Option 2: Node.js http-server
npx http-server -p 8000

# Buka browser
# http://localhost:8000
```

**No Build Process Required!** Langsung buka `index.html` di browser.

---

## Struktur Project

```
unumatch/
├── index.html              # Landing page dengan hero & features
├── survey.html             # Survey form dengan 3 steps
├── hasil.html              # Results page dengan visualisasi
├── dataset_unu.csv         # Training data (1001 records)
├── css/
│   └── style.css          # Custom styles & animations
├── js/
│   └── script.js          # Clustering logic & utilities
├── README.md              # Dokumentasi lengkap (this file)
└── QUICKSTART.md          # Quick setup guide
```



---

## Algoritma Machine Learning

### Pendekatan: Random Forest Classifier

UNU-Match menggunakan **Random Forest Machine Learning** untuk merekomendasikan jurusan. Random Forest adalah algoritma ensemble learning yang menggabungkan prediksi dari banyak decision trees untuk hasil yang lebih akurat dan robust.

### 1. Feature Extraction

**12 Features Input:**
- Nilai Mata Pelajaran (7): `mtk, inggris, agama, fisika, kimia, biologi, ekonomi` (0-100)
- Minat (4): `minat_teknik, minat_kesehatan, minat_bisnis, minat_pendidikan` (1-5)
- Hafalan (1): `hafalan` (0 = tidak, 1 = ya)

### 2. Normalization

Setiap feature dinormalisasi ke rentang 0-1 agar seimbang dalam perhitungan:
```javascript
// Nilai mata pelajaran: 0-100 → 0-1
normalized = value / 100

// Minat: 1-5 → 0-1
normalized = (value - 1) / 4

// Hafalan: sudah 0-1, tidak perlu normalisasi
```

### 3. Model Training (Python)

Model dilatih menggunakan **scikit-learn** dengan konfigurasi:
- **Algoritma**: Random Forest Classifier
- **Jumlah Trees**: 200 decision trees
- **Max Depth**: 10 (mencegah overfitting)
- **Min Samples Split**: 5
- **Training Data**: 1001 mahasiswa
- **Test Accuracy**: ~73.5%

```python
# Training di Python (offline)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)
```

### 4. Prediction & Results

Sistem mengembalikan 3 program studi dengan probability tertinggi, lengkap dengan:
- **Match Percentage**: Tingkat kecocokan (%)
- **Confidence Level**: Sangat Tinggi / Tinggi / Sedang / Rendah
- **Feature Contributions**: Penjelasan mengapa jurusan ini cocok

### 7. Feature Importance

Model menghitung kontribusi setiap fitur dalam prediksi:
- **minat_teknik**: 14.2% (paling penting)
- **minat_bisnis**: 10.8%
- **minat_kesehatan**: 9.5%
- **mtk**: 8.7%
- dll.

Informasi ini digunakan untuk menjelaskan hasil rekomendasi kepada user.


---

## Program Studi

### 10 Program Studi Tersedia

| No | Program Studi | Icon | Bidang |
|----|---------------|------|--------|
| 1 | S1 Informatika | `hgi-laptop` | Teknologi & Programming |
| 2 | S1 Farmasi | `hgi-test-tube` | Kesehatan & Obat-obatan |
| 3 | S1 Teknik Elektro | `hgi-flash` | Kelistrikan & Elektronika |
| 4 | S1 Agribisnis | `hgi-plant-02` | Bisnis Pertanian |
| 5 | S1 Akuntansi | `hgi-money-bag-02` | Keuangan & Audit |
| 6 | S1 Pendidikan Bahasa Inggris | `hgi-globe-02` | Pendidikan & Bahasa |
| 7 | S1 PGSD | `hgi-teacher` | Pendidikan Guru SD |
| 8 | S1 Studi Islam Interdisipliner | `hgi-book-open-02` | Kajian Keislaman |
| 9 | S1 Manajemen | `hgi-analytics-01` | Bisnis & Organisasi |
| 10 | S1 Teknologi Hasil Pertanian | `hgi-medicine-01` | Pengolahan Pangan |

---

## Customization

**Ubah Color Scheme**: Edit CSS variables di `css/style.css` (--primary-green, dll)

**Tambah Program Studi**: 
1. Update `dataset_unu.csv`
2. Edit `prodiInfo` object di `js/script.js`
3. Update cards di `index.html`

**Retrain Model**: Edit parameters di `ml/train_model.py` lalu jalankan training ulang


---

## Browser Support

Supports all modern browsers (Chrome, Firefox, Safari, Edge 90+)

**Requirements**: ES6+ JavaScript, LocalStorage, CSS Grid/Flexbox

---

## FAQ

### Q: Apakah hasil rekomendasi 100% akurat?
**A:** Hasil adalah rekomendasi berdasarkan data historis 1000+ mahasiswa. Akurasi tergantung pada kemiripan profil user dengan data training. Gunakan sebagai panduan, bukan keputusan final.

### Q: Data saya aman?
**A:** Ya! Semua data tersimpan di LocalStorage browser Anda. Tidak ada data yang dikirim ke server. Hapus browser cache untuk menghapus data.

### Q: Bisa offline?
**A:** Setelah halaman dimuat, semua proses berjalan di browser. Dataset juga sudah termuat. Namun, loading awal membutuhkan internet untuk CDN (TailwindCSS).

### Q: Bagaimana cara menghapus data survey?
**A:** Klik icon reset (⟳) di header halaman survey, atau clear LocalStorage di DevTools.

### Q: Bagaimana cara update dataset?
**A:** Replace `dataset_unu.csv` dengan dataset baru (format sama 13 kolom) dan reload halaman.

---

## Kontribusi

Kontribusi welcome! Fork repository → Create branch → Commit changes → Push → Open Pull Request

---

## Lisensi

**MIT License** - Copyright (c) 2026 UNU-Match

---

## Kontak

- **Email**: [ulilirvan@gmail.com]
- **GitHub Issues**: [Create an issue](https://github.com/ulililzam/unu-match/issues)

---

## Roadmap

**v1.1**: Export PDF, Email results, Analytics
**v1.2**: Admin dashboard, User accounts
**v2.0**: Backend API, Mobile apps

---

**Last Updated**: January 20, 2026  
**Version**: 1.0.0  
**Status**: Production Ready
