#!/usr/bin/env python3
"""
Generate Realistic UNU Dataset
Create more human-like student profiles with natural variations
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

def generate_realistic_student(prodi):
    """Generate realistic student profile based on program"""
    
    # Base profiles for each major with realistic patterns
    # KKM Indonesia = 75, tapi realitas banyak siswa 70-80
    profiles = {
        "S1 Informatika": {
            "mtk": (80, 95, 8),      # (mean, max, std) - High math for CS
            "inggris": (75, 90, 10),
            "agama": (75, 85, 8),    # KKM baseline
            "fisika": (78, 92, 9),   # High physics
            "kimia": (73, 85, 10),
            "biologi": (72, 82, 10),
            "ekonomi": (73, 83, 10),
            "minat_teknik": (4, 5, 0.5),  # STRONG tech interest
            "minat_kesehatan": (1, 3, 0.8),
            "minat_bisnis": (2, 4, 0.8),
            "minat_pendidikan": (1, 3, 0.8),
            "hafalan": 0.05  # 5% hafal (very low for tech)
        },
        "S1 Farmasi": {
            "mtk": (74, 86, 9),
            "inggris": (75, 87, 9),
            "agama": (76, 86, 8),
            "fisika": (73, 83, 9),
            "kimia": (82, 95, 7),   # STRONG chemistry
            "biologi": (81, 95, 7), # STRONG biology
            "ekonomi": (74, 84, 9),
            "minat_teknik": (1, 3, 0.8),
            "minat_kesehatan": (4, 5, 0.5),  # STRONG health interest
            "minat_bisnis": (2, 4, 0.8),
            "minat_pendidikan": (2, 4, 0.8),
            "hafalan": 0.1  # Moderate
        },
        "S1 Teknik Elektro": {
            "mtk": (81, 95, 8),     # High math
            "inggris": (74, 86, 9),
            "agama": (75, 85, 8),
            "fisika": (83, 96, 7),  # VERY HIGH physics
            "kimia": (74, 84, 9),
            "biologi": (72, 82, 9),
            "ekonomi": (73, 83, 9),
            "minat_teknik": (4, 5, 0.5),  # STRONG tech interest
            "minat_kesehatan": (1, 3, 0.8),
            "minat_bisnis": (2, 4, 0.8),
            "minat_pendidikan": (1, 3, 0.8),
            "hafalan": 0.05  # Very low
        },
        "S1 Agribisnis": {
            "mtk": (74, 84, 8),
            "inggris": (74, 84, 8),
            "agama": (76, 86, 8),
            "fisika": (73, 83, 9),
            "kimia": (75, 85, 8),
            "biologi": (78, 91, 8),  # Higher biology
            "ekonomi": (77, 90, 8),  # Higher economics
            "minat_teknik": (1, 3, 0.8),
            "minat_kesehatan": (2, 4, 0.8),
            "minat_bisnis": (3, 5, 0.7),  # Strong business
            "minat_pendidikan": (2, 4, 0.8),
            "hafalan": 0.08
        },
        "S1 Akuntansi": {
            "mtk": (77, 92, 8),     # High math for accounting
            "inggris": (74, 86, 9),
            "agama": (75, 85, 8),
            "fisika": (73, 83, 9),
            "kimia": (73, 83, 9),
            "biologi": (73, 83, 9),
            "ekonomi": (81, 95, 7),  # VERY HIGH economics
            "minat_teknik": (1, 3, 0.8),
            "minat_kesehatan": (1, 3, 0.8),
            "minat_bisnis": (4, 5, 0.5),  # STRONG business
            "minat_pendidikan": (2, 4, 0.8),
            "hafalan": 0.08
        },
        "S1 Pendidikan Bahasa Inggris": {
            "mtk": (74, 84, 8),
            "inggris": (82, 96, 6),  # VERY HIGH English
            "agama": (77, 87, 7),
            "fisika": (73, 83, 9),
            "kimia": (73, 83, 9),
            "biologi": (74, 84, 8),
            "ekonomi": (74, 84, 8),
            "minat_teknik": (1, 3, 0.8),
            "minat_kesehatan": (2, 4, 0.8),
            "minat_bisnis": (2, 4, 0.8),
            "minat_pendidikan": (4, 5, 0.5),  # STRONG education interest
            "hafalan": 0.12
        },
        "S1 PGSD": {
            "mtk": (75, 87, 8),
            "inggris": (75, 87, 8),
            "agama": (77, 91, 7),
            "fisika": (73, 83, 9),
            "kimia": (73, 83, 9),
            "biologi": (74, 84, 8),
            "ekonomi": (74, 84, 8),
            "minat_teknik": (1, 3, 0.8),
            "minat_kesehatan": (2, 4, 0.8),
            "minat_bisnis": (2, 4, 0.8),
            "minat_pendidikan": (4, 5, 0.5),  # STRONG education interest
            "hafalan": 0.15
        },
        "S1 Studi Islam Interdisipliner": {
            "mtk": (73, 83, 9),
            "inggris": (74, 84, 8),
            "agama": (85, 98, 5),    # VERY HIGH religious studies
            "fisika": (73, 83, 9),
            "kimia": (73, 83, 9),
            "biologi": (73, 83, 9),
            "ekonomi": (74, 84, 8),
            "minat_teknik": (1, 3, 0.8),
            "minat_kesehatan": (2, 4, 0.8),
            "minat_bisnis": (2, 4, 0.8),
            "minat_pendidikan": (3, 5, 0.7),  # Strong education interest
            "hafalan": 0.7  # 70% hafal (high but not automatic)
        },
        "S1 Manajemen": {
            "mtk": (75, 87, 8),
            "inggris": (76, 91, 7),  # High English for management
            "agama": (75, 85, 8),
            "fisika": (73, 83, 9),
            "kimia": (73, 83, 9),
            "biologi": (73, 83, 9),
            "ekonomi": (78, 95, 8),  # HIGH economics
            "minat_teknik": (2, 4, 0.8),
            "minat_kesehatan": (1, 3, 0.8),
            "minat_bisnis": (4, 5, 0.5),  # STRONG business
            "minat_pendidikan": (2, 4, 0.8),
            "hafalan": 0.08
        },
        "S1 Teknologi Hasil Pertanian": {
            "mtk": (74, 84, 8),
            "inggris": (74, 84, 8),
            "agama": (75, 85, 8),
            "fisika": (76, 87, 8),
            "kimia": (78, 91, 7),    # High chemistry
            "biologi": (78, 91, 7),  # High biology
            "ekonomi": (74, 84, 8),
            "minat_teknik": (3, 5, 0.7),     # Tech interest
            "minat_kesehatan": (3, 5, 0.7), # Health interest
            "minat_bisnis": (2, 4, 0.8),
            "minat_pendidikan": (2, 4, 0.8),
            "hafalan": 0.08
        }
    }
    
    profile = profiles[prodi]
    student = {}
    
    # Generate grades with realistic correlation
    # KKM Indonesia = 75, tapi realitas ada yang 65-70 juga masuk kuliah
    for subject in ["mtk", "inggris", "agama", "fisika", "kimia", "biologi", "ekonomi"]:
        mean, max_val, std = profile[subject]
        # Use truncated normal to keep values in reasonable range
        value = np.random.normal(mean, std)
        value = np.clip(value, 65, 100)  # Realistic: min 65 (di bawah KKM tapi masih masuk kuliah)
        value = int(round(value))
        student[subject] = value
    
    # Generate interests (1-5 scale)
    for interest in ["minat_teknik", "minat_kesehatan", "minat_bisnis", "minat_pendidikan"]:
        mean, max_val, std = profile[interest]
        value = np.random.normal(mean, std)
        value = np.clip(value, 1, 5)
        value = int(round(value))
        student[interest] = value
    
    # Hafalan (0 or 1)
    student["hafalan"] = 1 if np.random.random() < profile["hafalan"] else 0
    
    student["prodi"] = prodi
    
    return student

def add_realistic_variations(students):
    """Add realistic edge cases and variations"""
    variations = []
    
    # 1. Siswa pintar tapi minat beda (3%)
    n_mismatch = int(len(students) * 0.03)
    for _ in range(n_mismatch):
        # Math smart but prefer business
        student = generate_realistic_student("S1 Informatika")
        student["minat_teknik"] = np.random.randint(2, 4)  # Lower tech interest
        student["minat_bisnis"] = np.random.randint(4, 6)  # High business
        student["prodi"] = "S1 Akuntansi"
        student["hafalan"] = 0  # No hafalan
        variations.append(student)
    
    # 2. Siswa passion tinggi tapi nilai KKM pas-pasan (3%)
    for _ in range(n_mismatch):
        student = generate_realistic_student("S1 Informatika")
        # KKM-level grades but VERY high interest
        for subject in ["mtk", "fisika", "inggris"]:
            student[subject] = np.random.randint(75, 82)  # Around KKM
        student["minat_teknik"] = 5  # MAXIMUM interest
        student["hafalan"] = 0
        variations.append(student)
        student["minat_teknik"] = 5
        variations.append(student)
    
    # 3. Siswa serba bisa (1%)
    n_allrounder = int(len(students) * 0.01)
    for _ in range(n_allrounder):
        prodi = np.random.choice([s["prodi"] for s in students])
        student = generate_realistic_student(prodi)
        # High grades in everything
        for subject in ["mtk", "inggris", "agama", "fisika", "kimia", "biologi", "ekonomi"]:
            student[subject] = np.random.randint(88, 98)
        variations.append(student)
    
    # 4. Siswa KKM pas-pasan (2%)
    n_struggle = int(len(students) * 0.02)
    for _ in range(n_struggle):
        prodi = np.random.choice([s["prodi"] for s in students])
        student = generate_realistic_student(prodi)
        # KKM-level grades (realitas Indonesia: bisa masuk kuliah)
        for subject in ["mtk", "inggris", "agama", "fisika", "kimia", "biologi", "ekonomi"]:
            student[subject] = np.random.randint(70, 78)  # Around KKM
        variations.append(student)
    
    return variations

def generate_dataset(n_students=1000):
    """Generate complete realistic dataset"""
    
    # Target distribution (roughly equal)
    prodis = [
        "S1 Informatika",
        "S1 Farmasi",
        "S1 Teknik Elektro",
        "S1 Agribisnis",
        "S1 Akuntansi",
        "S1 Pendidikan Bahasa Inggris",
        "S1 PGSD",
        "S1 Studi Islam Interdisipliner",
        "S1 Manajemen",
        "S1 Teknologi Hasil Pertanian"
    ]
    
    students_per_prodi = n_students // len(prodis)
    
    students = []
    
    # Generate base students
    for prodi in prodis:
        for _ in range(students_per_prodi):
            student = generate_realistic_student(prodi)
            students.append(student)
    
    # Add realistic variations and edge cases
    variations = add_realistic_variations(students)
    students.extend(variations)
    
    # Shuffle to make it more natural
    np.random.shuffle(students)
    
    # Trim to exactly n_students
    students = students[:n_students]
    
    return students

def main():
    """Generate and save realistic dataset"""
    print("🎓 Generating Realistic UNU Student Dataset...")
    print("=" * 60)
    
    # Generate dataset
    students = generate_dataset(n_students=1000)
    
    # Convert to DataFrame
    df = pd.DataFrame(students)
    
    # Reorder columns
    column_order = [
        "mtk", "inggris", "agama", "fisika", "kimia", 
        "biologi", "ekonomi", "minat_teknik", "minat_kesehatan",
        "minat_bisnis", "minat_pendidikan", "hafalan", "prodi"
    ]
    df = df[column_order]
    
    # Statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"Total students: {len(df)}")
    print(f"\n📋 Class Distribution:")
    print(df['prodi'].value_counts().sort_index())
    
    print(f"\n📈 Grade Statistics:")
    grade_cols = ["mtk", "inggris", "agama", "fisika", "kimia", "biologi", "ekonomi"]
    print(df[grade_cols].describe().round(1))
    
    print(f"\n⭐ Interest Statistics:")
    interest_cols = ["minat_teknik", "minat_kesehatan", "minat_bisnis", "minat_pendidikan"]
    print(df[interest_cols].describe().round(2))
    
    print(f"\n📖 Hafalan Distribution:")
    print(df['hafalan'].value_counts())
    
    # Save to CSV
    output_path = Path('../dataset_unu.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Dataset saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")
    
    print("\n" + "=" * 60)
    print("🎉 Realistic dataset generation complete!")
    print("\n💡 Key improvements:")
    print("   ✅ Natural grade variations (not all 70-80)")
    print("   ✅ Logical correlations (high math → Informatika)")
    print("   ✅ Edge cases (smart but different interest)")
    print("   ✅ Realistic distributions")
    print("   ✅ Human-like patterns")
    
    print("\n🔄 Next step: Retrain model with new data")
    print("   cd ml && python train_model.py")

if __name__ == "__main__":
    main()
