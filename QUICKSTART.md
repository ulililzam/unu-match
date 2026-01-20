# Quick Start - UNU-Match

Panduan cepat untuk menjalankan UNU-Match dalam 3 langkah!

---

## 3 Langkah Setup

### 1. Download Project

```bash
# Option A: Clone from Git
git clone https://github.com/yourusername/unumatch.git
cd unumatch

# Option B: Download ZIP
# Extract ZIP file ke folder pilihan Anda
```

### 2. Jalankan Local Server

**Option A: Python** (Recommended)
```bash
python3 -m http.server 8000
```

**Option B: Node.js**
```bash
npx http-server -p 8000
```

**Option C: PHP**
```bash
php -S localhost:8000
```

**Option D: Langsung Buka File** (Tanpa Server)
```
Double-click index.html
```
Note: Beberapa fitur (seperti loading CSV) mungkin tidak berfungsi tanpa server.

### 3. Buka di Browser

```
http://localhost:8000
```

**That's it!** Aplikasi sudah siap digunakan.

---

## Checklist Lengkap

Pastikan semua file ada di folder project:

```
index.html
survey.html
hasil.html
dataset_unu.csv
css/style.css
js/script.js
```

**CDN Dependencies:**
- TailwindCSS: `https://cdn.tailwindcss.com`
- Hugeicons: `https://cdn.hugeicons.com/font/hgi-stroke-rounded.css`

---

## User Flow

1. **Landing Page** (`index.html`)
   - Lihat fitur dan informasi
   - Klik "Mulai Sekarang"

2. **Survey** (`survey.html`)
   - **Step 1**: Isi nilai 7 mata pelajaran
   - **Step 2**: Rating minat 4 bidang
   - **Step 3**: Toggle hafalan
   - Klik "Lihat Hasil"

3. **Hasil** (`hasil.html`)
   - Lihat 3 rekomendasi jurusan
   - Cek match percentage & detail
   - Cetak atau bagikan hasil

---

## Testing Scenario

### Test Case 1: Profil Mahasiswa IT
```
Nilai:
- Matematika: 90
- Inggris: 85
- Agama: 70
- Fisika: 80
- Kimia: 65
- Biologi: 60
- Ekonomi: 70

Minat:
- Teknik: 5 bintang
- Kesehatan: 2 bintang
- Bisnis: 3 bintang
- Pendidikan: 2 bintang

Hafalan: Tidak

Expected: Informatika / Teknik Elektro
```

### Test Case 2: Profil Mahasiswa Kesehatan
```
Nilai:
- Matematika: 70
- Inggris: 75
- Agama: 80
- Fisika: 65
- Kimia: 90
- Biologi: 95
- Ekonomi: 70

Minat:
- Teknik: 2 bintang
- Kesehatan: 5 bintang
- Bisnis: 2 bintang
- Pendidikan: 3 bintang

Hafalan: Tidak

Expected: Farmasi / Teknologi Hasil Pertanian
```

### Test Case 3: Profil Mahasiswa Bisnis
```
Nilai:
- Matematika: 85
- Inggris: 80
- Agama: 75
- Fisika: 65
- Kimia: 60
- Biologi: 70
- Ekonomi: 95

Minat:
- Teknik: 2 bintang
- Kesehatan: 2 bintang
- Bisnis: 5 bintang
- Pendidikan: 2 bintang

Hafalan: Tidak

Expected: Akuntansi / Manajemen
```

### Test Case 4: Profil Mahasiswa Pendidikan
```
Nilai:
- Matematika: 75
- Inggris: 90
- Agama: 85
- Fisika: 70
- Kimia: 65
- Biologi: 70
- Ekonomi: 70

Minat:
- Teknik: 2 bintang
- Kesehatan: 2 bintang
- Bisnis: 3 bintang
- Pendidikan: 5 bintang

Hafalan: Tidak

Expected: Pendidikan Bahasa Inggris / PGSD
```

### Test Case 5: Profil Mahasiswa Islam
```
Nilai:
- Matematika: 70
- Inggris: 75
- Agama: 95
- Fisika: 65
- Kimia: 70
- Biologi: 65
- Ekonomi: 70

Minat:
- Teknik: 2 bintang
- Kesehatan: 3 bintang
- Bisnis: 2 bintang
- Pendidikan: 3 bintang

Hafalan: Ya

Expected: Studi Islam Interdisipliner
```

---

## Troubleshooting

### Problem: CSV tidak termuat
**Solution**: 
```
- Pastikan menjalankan local server (jangan langsung buka file)
- Check console browser (F12) untuk error
- Verifikasi dataset_unu.csv ada di root folder
```

### Problem: Hasil tidak muncul
**Solution**:
```
- Clear LocalStorage (DevTools > Application > Local Storage > Clear)
- Refresh halaman
- Isi ulang survey
```

### Problem: Styling tidak muncul
**Solution**:
```
- Check internet connection (TailwindCSS CDN)
- Verifikasi css/style.css ada dan termuat
- Clear browser cache
```

### Problem: JavaScript error
**Solution**:
```
- Check console (F12) untuk detail error
- Verifikasi js/script.js ada dan termuat
- Pastikan browser support ES6+
```

---

## Mobile Testing

Test di berbagai device:

1. **Chrome DevTools**
   - F12 > Toggle Device Toolbar (Ctrl+Shift+M)
   - Test: iPhone 12, iPad, Galaxy S20

2. **Actual Devices**
   - Akses `http://YOUR_LOCAL_IP:8000` dari HP
   - Example: `http://192.168.1.10:8000`

3. **Check Responsiveness**
   - Slider mudah digeser
   - Star rating mudah diklik
   - Buttons tidak terlalu kecil
   - Text readable tanpa zoom

---

## Quick Customization

### Ubah Warna Hijau ke Biru

Edit `css/style.css`:
```css
:root {
    --primary-green: #3b82f6;      /* Biru */
    --primary-green-dark: #2563eb; /* Biru tua */
    --primary-green-light: #60a5fa; /* Biru muda */
    --accent-green: #dbeafe;       /* Biru sangat muda */
}
```

### Ubah Font

Edit `css/style.css`:
```css
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

body {
    font-family: 'Poppins', sans-serif;
}
```

### Ubah Logo/Title

Edit `index.html`, `survey.html`, `hasil.html`:
```html
<span class="text-xl font-bold text-dark">
    Your-<span class="text-green">Brand</span>
</span>
```

---

## Performance Check

### Lighthouse Score Target

```
Performance: 90+
Accessibility: 95+
Best Practices: 100
SEO: 90+
```

### Run Lighthouse

```bash
# Chrome DevTools
F12 > Lighthouse > Generate Report
```

---

## Deployment Options

### Option 1: GitHub Pages
```bash
# Push ke GitHub
git add .
git commit -m "Initial commit"
git push origin main

# Settings > Pages > Deploy from main branch
```

### Option 2: Netlify
```bash
# Drag & drop folder ke netlify.com
# atau connect ke GitHub repo
```

### Option 3: Vercel
```bash
# Import project dari GitHub
# atau deploy via CLI
npx vercel
```

### Option 4: cPanel/Shared Hosting
```
- Upload semua file via FTP
- Set index.html sebagai default page
```

---

## Pro Tips

1. **Clear Cache**: Selalu clear cache saat testing perubahan
2. **Use Incognito**: Test di incognito mode untuk fresh state
3. **Check Console**: Monitor console untuk errors/warnings
4. **Mobile First**: Test mobile view dulu sebelum desktop
5. **Real Data**: Test dengan data real user untuk validasi

---

## Butuh Bantuan?

- Baca [README.md](README.md) lengkap
- Report bug via GitHub Issues
- Diskusi di GitHub Discussions
- Email: your-email@example.com

---

## Verification Checklist

Sebelum deploy, pastikan:

- [ ] Semua pages load tanpa error
- [ ] Survey form bisa diisi lengkap
- [ ] Hasil muncul dengan benar
- [ ] Print function bekerja
- [ ] Share function bekerja
- [ ] Responsive di mobile
- [ ] No console errors
- [ ] Dataset termuat dengan benar
- [ ] LocalStorage bekerja
- [ ] Reset function bekerja

---

**Happy Testing!**

Jika ada pertanyaan, jangan ragu untuk bertanya!

---

**Setup Time**: < 5 minutes  
**Difficulty**: Beginner-friendly  
**No Build Required**: Just open & run!
