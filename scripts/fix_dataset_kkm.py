#!/usr/bin/env python3
"""
Dataset Fixer - Adjust nilai to realistic KKM 70 standard
Shifts grade distribution to match Indonesian high school reality
"""

import csv
import random
from datetime import datetime

print("="*70)
print("UNU-MATCH DATASET FIXER - KKM 70 ADJUSTMENT")
print("="*70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Configuration
MAPEL_COLUMNS = ['mtk', 'inggris', 'agama', 'fisika', 'kimia', 'biologi', 'ekonomi']
SHIFT_AMOUNT = 10  # Add 10 points to make realistic
ADD_VARIANCE = True  # Add slight randomness for natural feel

def fix_nilai(nilai_str, add_variance=True):
    """
    Fix nilai to be more realistic with KKM 70
    - Base: +10 points
    - Variance: ±2 points for natural feel
    - Cap: Maximum 100
    """
    nilai = int(nilai_str)
    
    # Add base shift
    nilai_new = nilai + SHIFT_AMOUNT
    
    # Add slight variance for natural distribution
    if add_variance:
        variance = random.randint(-2, 2)
        nilai_new += variance
    
    # Cap at 100
    nilai_new = min(100, nilai_new)
    
    # Floor at 50 (minimum possible)
    nilai_new = max(50, nilai_new)
    
    return nilai_new

# Read original dataset
print("📖 Reading original dataset...")
with open('dataset_unu.csv', 'r') as f:
    reader = csv.DictReader(f)
    data_original = list(reader)

print(f"   ✓ Loaded {len(data_original)} records")

# Analyze original
print("\n📊 ORIGINAL DATASET STATISTICS:")
print("-" * 70)
all_original = []
for m in MAPEL_COLUMNS:
    values = [int(row[m]) for row in data_original]
    all_original.extend(values)
    mean = sum(values) / len(values)
    below_70 = sum(1 for v in values if v < 70)
    print(f"   {m.upper():10s} - Mean: {mean:.2f}, <70: {below_70} ({below_70/len(values)*100:.1f}%)")

mean_orig = sum(all_original) / len(all_original)
below_70_orig = sum(1 for v in all_original if v < 70)
print(f"\n   OVERALL    - Mean: {mean_orig:.2f}, <70: {below_70_orig} ({below_70_orig/len(all_original)*100:.1f}%)")

# Fix dataset
print("\n🔧 Fixing dataset...")
print(f"   Strategy: Shift +{SHIFT_AMOUNT} points with ±2 variance")

data_fixed = []
for row in data_original:
    new_row = {}
    for col in row.keys():
        if col in MAPEL_COLUMNS:
            # Fix nilai mapel
            new_row[col] = fix_nilai(row[col], ADD_VARIANCE)
        else:
            # Keep other columns (minat, hafalan, prodi)
            new_row[col] = row[col]
    data_fixed.append(new_row)

print(f"   ✓ Fixed {len(data_fixed)} records")

# Analyze fixed
print("\n📊 FIXED DATASET STATISTICS:")
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
print(f"   Mean improved: {mean_orig:.2f} → {mean_fixed:.2f} (+{mean_fixed - mean_orig:.2f})")
print(f"   Below KKM 70: {below_70_orig/len(all_original)*100:.1f}% → {below_70_fixed/len(all_fixed)*100:.1f}% ({below_70_orig/len(all_original)*100 - below_70_fixed/len(all_fixed)*100:.1f}% reduction)")
print(f"   Above KKM 70: {above_70_fixed/len(all_fixed)*100:.1f}% (was {(len(all_original)-below_70_orig)/len(all_original)*100:.1f}%)")

# Realism check
print("\n🎯 REALISM CHECK:")
print("-" * 70)
if mean_fixed >= 75 and mean_fixed <= 80:
    print("   ✅ Mean is REALISTIC (75-80 expected for Indonesian rapor)")
elif mean_fixed >= 72 and mean_fixed < 75:
    print("   ⚠️  Mean is ACCEPTABLE (slightly low but okay)")
elif mean_fixed >= 80:
    print("   ⚠️  Mean is SLIGHTLY HIGH (but acceptable)")
else:
    print("   ❌ Mean is TOO LOW (need more adjustment)")

if below_70_fixed/len(all_fixed)*100 <= 20:
    print("   ✅ Below KKM percentage is REALISTIC (10-20% expected)")
elif below_70_fixed/len(all_fixed)*100 <= 30:
    print("   ⚠️  Below KKM percentage is ACCEPTABLE (slightly high)")
else:
    print("   ❌ Below KKM percentage is TOO HIGH (need more adjustment)")

if above_85_fixed/len(all_fixed)*100 >= 15 and above_85_fixed/len(all_fixed)*100 <= 25:
    print("   ✅ High achievers (>=85) percentage is REALISTIC (15-25% expected)")
else:
    print(f"   ⚠️  High achievers: {above_85_fixed/len(all_fixed)*100:.1f}% (expected 15-25%)")

# Backup original
print("\n💾 Backing up original dataset...")
backup_name = f"dataset_unu_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(backup_name, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=data_original[0].keys())
    writer.writeheader()
    writer.writerows(data_original)
print(f"   ✓ Backup saved: {backup_name}")

# Save fixed dataset
print("\n💾 Saving fixed dataset...")
with open('dataset_unu.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=data_fixed[0].keys())
    writer.writeheader()
    writer.writerows(data_fixed)
print(f"   ✓ Fixed dataset saved: dataset_unu.csv")

# Show sample comparison
print("\n📋 SAMPLE COMPARISON (First 5 records):")
print("-" * 70)
for i in range(min(5, len(data_original))):
    print(f"\nRecord {i+1}:")
    print(f"   Original: Mtk={data_original[i]['mtk']}, Ing={data_original[i]['inggris']}, Bio={data_original[i]['biologi']}")
    print(f"   Fixed:    Mtk={data_fixed[i]['mtk']}, Ing={data_fixed[i]['inggris']}, Bio={data_fixed[i]['biologi']}")
    print(f"   Prodi: {data_fixed[i]['prodi']}")

print("\n" + "="*70)
print("✅ DATASET FIXED SUCCESSFULLY!")
print("="*70)
print("\n📝 NEXT STEPS:")
print("   1. Review the fixed dataset statistics above")
print("   2. Run: cd ml && python3 train_model.py")
print("   3. Check new model accuracy")
print("   4. Deploy updated model to production")
print(f"\n💡 Backup location: {backup_name}")
print("   (You can restore if needed)")
print()
