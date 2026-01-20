# UNU-Match - Sistem Rekomendasi Jurusan Mahasiswa

![UNU-Match](https://img.shields.io/badge/Version-1.0.0-green)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black)

Sistem cerdas berbasis web yang membantu calon mahasiswa Universitas Nahdlatul Ulama menemukan program studi yang paling cocok berdasarkan nilai akademik, minat, dan preferensi belajar.

---

## Daftar Isi

- [Tentang Project](#tentang-project)
- [Fitur Utama](#fitur-utama)
- [Teknologi](#teknologi)
- [Cara Menggunakan](#cara-menggunakan)
- [Struktur Project](#struktur-project)
- [Algoritma Clustering](#algoritma-clustering)
- [Program Studi](#program-studi)
- [Customization](#customization)
- [Browser Support](#browser-support)
- [FAQ](#faq)
- [Kontribusi](#kontribusi)
- [Lisensi](#lisensi)

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

### Core Technologies
- **HTML5**: Semantic markup & modern web standards
- **CSS3**: Custom properties, Flexbox, Grid, Animations
- **Vanilla JavaScript (ES6+)**: No frameworks, pure performance

### Libraries & Tools
- **TailwindCSS CDN**: Utility-first CSS framework
- **Hugeicons CDN**: 4,600+ professional stroke-rounded icons
- **Google Fonts (Inter)**: Modern, readable typography
- **LocalStorage API**: Client-side data persistence
- **Web Share API**: Native sharing capabilities

### Design System
- **Color Palette**: 
  - Primary: `#16a34a` (Green)
  - Accent: `#dcfce7` (Light Green)
  - Background: `#f9fafb` (Gray)
- **Typography**: Inter font family (300-800 weights)
- **Icons**: Hugeicons stroke-rounded icon font
- **Shadows**: Layered shadow system for depth

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

### File Descriptions

#### HTML Files
- **index.html**: Landing page dengan call-to-action
- **survey.html**: Multi-step form dengan progress bar
- **hasil.html**: Displaying recommendations with details

#### CSS
- **style.css**: 
  - CSS Variables untuk theming
  - Custom components (btn, card, slider, rating)
  - Animations & transitions
  - Responsive breakpoints
  - Print styles

#### JavaScript
- **script.js**:
  - Dataset loading & parsing
  - K-NN clustering algorithm
  - LocalStorage utilities
  - UI helper functions
  - Form validation

#### Data
- **dataset_unu.csv**:
  - 1001 rows × 13 columns
  - 12 input features + 1 output (prodi)
  - Clean, normalized data

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

### 4. Tree Voting System

Setiap tree memberikan satu vote untuk program studi tertentu:
```javascript
// 200 trees voting
Tree 1 → Informatika
Tree 2 → Informatika  
Tree 3 → Farmasi
...
Tree 200 → Informatika

// Votes count:
// Informatika: 142 votes (71%)
// Farmasi: 38 votes (19%)
// Teknik Elektro: 20 votes (10%)
```

### 5. Probability Calculation

Probabilitas dihitung dari jumlah votes:
```javascript
probability = votes / total_trees
matchPercentage = probability × 100

// Contoh:
// Informatika: 142/200 = 0.71 → 71% match
```

### 6. Top 3 Recommendations

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

### Karakteristik Dataset

**Distribution Analysis:**
- Informatika: ~10% (High math & English)
- Farmasi: ~10% (High chemistry & biology)
- Teknik Elektro: ~10% (High physics & math)
- Agribisnis: ~10% (High biology & economy)
- Akuntansi: ~10% (High economy & business interest)
- Pendidikan Bahasa Inggris: ~10% (High English & education interest)
- PGSD: ~10% (High education interest)
- Studi Islam: ~10% (High religion & hafalan)
- Manajemen: ~10% (High economy & business interest)
- Teknologi Hasil Pertanian: ~10% (High biology & chemistry)

---

## Customization

### Ubah Color Scheme

Edit `css/style.css`:
```css
:root {
    --primary-green: #16a34a;       /* Ubah ke warna utama */
    --primary-green-dark: #15803d;  /* Versi lebih gelap */
    --primary-green-light: #22c55e; /* Versi lebih terang */
    --accent-green: #dcfce7;        /* Background accent */
}
```

### Ubah Font

Edit `css/style.css`:
```css
@import url('https://fonts.googleapis.com/css2?family=YourFont:wght@300;400;600;700&display=swap');

body {
    font-family: 'YourFont', sans-serif;
}
```

### Tambah Program Studi Baru

1. **Update Dataset**: Tambah data ke `dataset_unu.csv`
2. **Update Info**: Edit `js/script.js` di object `prodiInfo`
```javascript
const prodiInfo = {
    "S1 Program Baru": {
        name: "Program Baru",
        fullName: "S1 Program Baru",
        description: "Deskripsi singkat...",
        subjects: ["Mata Kuliah 1", "Mata Kuliah 2"],
        careers: ["Karir 1", "Karir 2"]
    }
};
```
3. **Update Landing**: Edit `index.html` untuk menambah card prodi baru

### Adjust Model Configuration

Model Random Forest dilatih di Python. Untuk mengubah parameter model:
```python
# Edit scripts/train_model.py
model = RandomForestClassifier(
    n_estimators=200,    # Lebih banyak trees = lebih akurat tapi lebih lambat
    max_depth=10,        # Kedalaman tree (mencegah overfitting)
    min_samples_split=5, # Minimum samples untuk split
    random_state=42
)
```

Setelah training ulang, export model ke JSON dengan:
```bash
python scripts/export_model.py
```


---

## Browser Support

| Browser | Version | Status |
|---------|---------|--------|
| Chrome | 90+ | Full Support |
| Firefox | 88+ | Full Support |
| Safari | 14+ | Full Support |
| Edge | 90+ | Full Support |
| Opera | 76+ | Full Support |

**Required Browser Features:**
- ES6+ JavaScript
- LocalStorage API
- CSS Grid & Flexbox
- CSS Custom Properties
- HTML5 Semantic Elements
- Web Font Loading (for Hugeicons)

**Optional Features:**
- Web Share API (fallback to clipboard)
- Print API

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

### Q: Kenapa hanya 3 rekomendasi?
**A:** Berdasarkan research, 3 pilihan adalah sweet spot antara memberikan options dan menghindari decision paralysis.

### Q: Bisa ubah jumlah rekomendasi?
**A:** Ya, edit `js/script.js` di function `predictProdi()`:
```javascript
const top3 = matches.slice(0, 3); // Ubah 3 menjadi 5 untuk top 5
```

### Q: Bagaimana cara update dataset?
**A:** Replace `dataset_unu.csv` dengan dataset baru. Format harus sama (13 kolom). Reload halaman untuk re-parse.

---

## Kontribusi

Kontribusi sangat welcome! Berikut cara berkontribusi:

### How to Contribute

1. **Fork** repository ini
2. **Create branch**: `git checkout -b feature/AmazingFeature`
3. **Commit changes**: `git commit -m 'Add some AmazingFeature'`
4. **Push to branch**: `git push origin feature/AmazingFeature`
5. **Open Pull Request**

### Contribution Ideas

- Bug fixes
- New features (e.g., export PDF)
- Documentation improvements
- UI/UX enhancements
- Testing & validation
- Localization (English version)
- Data visualization improvements

---

## 📄 Lisensi

**MIT License**

Copyright (c) 2026 UNU-Match

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Acknowledgments

- **Universitas Nahdlatul Ulama** - Untuk data dan konteks
- **TailwindCSS** - Utility-first CSS framework
- **Google Fonts** - Inter font family
- **Open Source Community** - Untuk inspirasi dan resources

---

## Kontak

Untuk pertanyaan, saran, atau bug report:

- **Email**: [your-email@example.com]
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/unumatch/issues)

---

## Roadmap

### Version 1.1 (Q2 2026)
- [ ] Export results to PDF
- [ ] Email notification results
- [ ] More detailed analytics
- [ ] Comparison mode (compare 2 prodi)

### Version 1.2 (Q3 2026)
- [ ] Admin dashboard for dataset management
- [ ] User accounts & history
- [ ] Social features (reviews, ratings)
- [ ] Advanced filtering

### Version 2.0 (Q4 2026)
- [ ] Backend API integration
- [ ] Real-time collaboration
- [ ] Mobile apps (iOS & Android)
- [ ] AI-powered career counseling

---

**Made with love for Universitas Nahdlatul Ulama students**

**Star this repo if you find it helpful!**

---

**Last Updated**: January 20, 2026  
**Version**: 1.0.0  
**Status**: Production Ready
