#!/usr/bin/env python3
"""
Dataset Fixer V2 - Stronger KKM adjustment
Target: <20% below KKM (more realistic)
"""

import csv
import random
from datetime import datetime

print("="*70)
print("UNU-MATCH DATASET FIXER V2 - STRONGER KKM ADJUSTMENT")
print("="*70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

MAPEL_COLUMNS = ['mtk', 'inggris', 'agama', 'fisika', 'kimia', 'biologi', 'ekonomi']

def fix_nilai_v2(nilai_str):
    """
    Stronger fix for realistic KKM distribution
    Strategy: 
    - Very low (<60): +15 points
    - Low (60-69): +12 points
    - Medium (70-79): +8 points
    - High (80-89): +5 points
    - Very high (90+): +3 points (keep excellence)
    """
    nilai = int(nilai_str)
    
    if nilai < 60:
        shift = random.randint(13, 17)  # +15 ±2
    elif nilai < 70:
        shift = random.randint(10, 14)  # +12 ±2
    elif nilai < 80:
        shift = random.randint(6, 10)   # +8 ±2
    elif nilai < 90:
        shift = random.randint(3, 7)    # +5 ±2
    else:
        shift = random.randint(1, 5)    # +3 ±2
    
    nilai_new = nilai + shift
    nilai_new = min(100, nilai_new)
    nilai_new = max(50, nilai_new)
    
    return nilai_new

# Read current dataset
print("📖 Reading current dataset...")
with open('dataset_unu.csv', 'r') as f:
    reader = csv.DictReader(f)
    data_current = list(reader)

print(f"   ✓ Loaded {len(data_current)} records")

# Analyze current
print("\n📊 CURRENT DATASET STATISTICS:")
print("-" * 70)
all_current = []
for m in MAPEL_COLUMNS:
    values = [int(row[m]) for row in data_current]
    all_current.extend(values)
    mean = sum(values) / len(values)
    below_70 = sum(1 for v in values if v < 70)
    print(f"   {m.upper():10s} - Mean: {mean:.2f}, <70: {below_70} ({below_70/len(values)*100:.1f}%)")

mean_curr = sum(all_current) / len(all_current)
below_70_curr = sum(1 for v in all_current if v < 70)
print(f"\n   OVERALL    - Mean: {mean_curr:.2f}, <70: {below_70_curr} ({below_70_curr/len(all_current)*100:.1f}%)")

# Check if already good enough
if below_70_curr/len(all_current)*100 <= 20 and mean_curr >= 75:
    print("\n✅ Dataset already in good shape! No need for further adjustment.")
    print(f"   Mean: {mean_curr:.2f} (target: 75-80)")
    print(f"   <KKM: {below_70_curr/len(all_current)*100:.1f}% (target: <20%)")
    exit(0)

# Restore from backup and apply stronger fix
print("\n🔧 Dataset needs stronger adjustment...")
print("   Loading backup for fresh fix...")

# Find latest backup
import glob
backups = glob.glob('dataset_unu_backup_*.csv')
if not backups:
    print("   ❌ No backup found! Using current dataset as base.")
    data_base = data_current
else:
    latest_backup = max(backups)
    print(f"   ✓ Using backup: {latest_backup}")
    with open(latest_backup, 'r') as f:
        reader = csv.DictReader(f)
        data_base = list(reader)

# Apply stronger fix
print("\n🔧 Applying stronger fix (adaptive shift based on nilai level)...")
data_fixed = []
for row in data_base:
    new_row = {}
    for col in row.keys():
        if col in MAPEL_COLUMNS:
            new_row[col] = fix_nilai_v2(row[col])
        else:
            new_row[col] = row[col]
    data_fixed.append(new_row)

print(f"   ✓ Fixed {len(data_fixed)} records")

# Analyze fixed
print("\n📊 IMPROVED DATASET STATISTICS:")
print("-" * 70)
all_fixed = []
for m in MAPEL_COLUMNS:
    values = [int(row[m]) for row in data_fixed]
    all_fixed.extend(values)
    mean = sum(values) / len(values)
    below_70 = sum(1 for v in values if v < 70)
    above_85 = sum(1 for v in values if v >= 85)
    print(f"   {m.upper():10s} - Mean: {mean:.2f}, <70: {below_70} ({below_70/len(values)*100:.1f}%), >=85: {above_85} ({above_85/len(values)*100:.1f}%)")

mean_fixed = sum(all_fixed) / len(all_fixed)
below_70_fixed = sum(1 for v in all_fixed if v < 70)
above_70_fixed = sum(1 for v in all_fixed if v >= 70)
above_85_fixed = sum(1 for v in all_fixed if v >= 85)

print(f"\n   OVERALL    - Mean: {mean_fixed:.2f}")
print(f"                <70: {below_70_fixed} ({below_70_fixed/len(all_fixed)*100:.1f}%)")
print(f"                >=70: {above_70_fixed} ({above_70_fixed/len(all_fixed)*100:.1f}%)")
print(f"                >=85: {above_85_fixed} ({above_85_fixed/len(all_fixed)*100:.1f}%)")

# Improvement summary
print("\n✨ IMPROVEMENT SUMMARY:")
print("-" * 70)
print(f"   Mean: {mean_curr:.2f} → {mean_fixed:.2f} (+{mean_fixed - mean_curr:.2f})")
print(f"   Below KKM 70: {below_70_curr/len(all_current)*100:.1f}% → {below_70_fixed/len(all_fixed)*100:.1f}%")
print(f"   Above KKM 70: {above_70_fixed/len(all_fixed)*100:.1f}%")

# Realism check
print("\n🎯 REALISM CHECK:")
print("-" * 70)

realism_score = 0
total_checks = 3

if mean_fixed >= 75 and mean_fixed <= 80:
    print("   ✅ Mean is REALISTIC (75-80 for Indonesian rapor)")
    realism_score += 1
elif mean_fixed >= 72 and mean_fixed < 75:
    print("   ⚠️  Mean is ACCEPTABLE (slightly low)")
    realism_score += 0.5
elif mean_fixed > 80 and mean_fixed <= 85:
    print("   ⚠️  Mean is ACCEPTABLE (slightly high)")
    realism_score += 0.5
else:
    print(f"   ❌ Mean needs adjustment (current: {mean_fixed:.2f})")

if below_70_fixed/len(all_fixed)*100 <= 20:
    print(f"   ✅ Below KKM is REALISTIC ({below_70_fixed/len(all_fixed)*100:.1f}% ≤ 20%)")
    realism_score += 1
elif below_70_fixed/len(all_fixed)*100 <= 30:
    print(f"   ⚠️  Below KKM is ACCEPTABLE ({below_70_fixed/len(all_fixed)*100:.1f}%)")
    realism_score += 0.5
else:
    print(f"   ❌ Below KKM too high ({below_70_fixed/len(all_fixed)*100:.1f}%)")

if above_85_fixed/len(all_fixed)*100 >= 15 and above_85_fixed/len(all_fixed)*100 <= 30:
    print(f"   ✅ High achievers realistic ({above_85_fixed/len(all_fixed)*100:.1f}%)")
    realism_score += 1
else:
    print(f"   ⚠️  High achievers: {above_85_fixed/len(all_fixed)*100:.1f}% (expected 15-30%)")
    if above_85_fixed/len(all_fixed)*100 >= 10 and above_85_fixed/len(all_fixed)*100 <= 35:
        realism_score += 0.5

print(f"\n   📊 Realism Score: {realism_score:.1f}/{total_checks}")

if realism_score >= 2.5:
    print("   🎉 EXCELLENT! Dataset is highly realistic!")
elif realism_score >= 2:
    print("   ✅ GOOD! Dataset is realistic enough for production")
elif realism_score >= 1:
    print("   ⚠️  FAIR. Dataset is usable but could be better")
else:
    print("   ❌ POOR. Dataset needs more adjustment")

# Save fixed dataset
print("\n💾 Saving improved dataset...")
with open('dataset_unu.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=data_fixed[0].keys())
    writer.writeheader()
    writer.writerows(data_fixed)
print(f"   ✓ Improved dataset saved: dataset_unu.csv")

# Sample comparison
print("\n📋 SAMPLE HASIL (First 3 records):")
print("-" * 70)
for i in range(min(3, len(data_base))):
    print(f"\nSiswa {i+1} - {data_fixed[i]['prodi']}:")
    print(f"   Before: Mtk={data_base[i]['mtk']}, Ing={data_base[i]['inggris']}, Fis={data_base[i]['fisika']}")
    print(f"   After:  Mtk={data_fixed[i]['mtk']}, Ing={data_fixed[i]['inggris']}, Fis={data_fixed[i]['fisika']}")
    
    # Count how many below KKM
    before_below = sum(1 for m in MAPEL_COLUMNS if int(data_base[i][m]) < 70)
    after_below = sum(1 for m in MAPEL_COLUMNS if int(data_fixed[i][m]) < 70)
    print(f"   Below KKM: {before_below}/7 → {after_below}/7 mapel")

print("\n" + "="*70)
print("✅ DATASET IMPROVEMENT COMPLETE!")
print("="*70)
print("\n📝 NEXT STEPS:")
print("   1. ✅ Dataset fixed successfully")
print("   2. ⏭️  Retrain model: cd ml && python3 train_model.py")
print("   3. 📊 Compare old vs new model accuracy")
print("   4. 🚀 Deploy if accuracy improved")
print()
