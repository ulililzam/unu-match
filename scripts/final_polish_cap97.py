#!/usr/bin/env python3
"""
Final Polish - Cap nilai maksimal di 97 (no perfect 100)
"""

import csv

print("="*70)
print("FINAL POLISH - Cap Nilai Maksimal 97")
print("="*70)

mapel = ['mtk', 'inggris', 'agama', 'fisika', 'kimia', 'biologi', 'ekonomi']

# Read current dataset
with open('dataset_unu.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

print(f"\n📖 Loaded {len(data)} records")

# Count nilai 100
count_100 = 0
for row in data:
    for m in mapel:
        if int(row[m]) == 100:
            count_100 += 1

print(f"📊 Found {count_100} nilai with perfect 100")

# Cap at 97
data_capped = []
changed_count = 0

for row in data:
    new_row = {}
    for col in row.keys():
        if col in mapel:
            nilai = int(row[col])
            if nilai == 100:
                # Random 95-97 for natural feel
                import random
                new_row[col] = random.randint(95, 97)
                changed_count += 1
            elif nilai == 99:
                # Also cap 99 to 95-97
                new_row[col] = random.randint(95, 97)
                changed_count += 1
            elif nilai == 98:
                # Cap 98 to 95-97
                new_row[col] = random.randint(95, 97)
                changed_count += 1
            else:
                new_row[col] = nilai
        else:
            new_row[col] = row[col]
    data_capped.append(new_row)

print(f"🔧 Capped {changed_count} nilai from 98-100 to 95-97")

# Save
with open('dataset_unu.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=data_capped[0].keys())
    writer.writeheader()
    writer.writerows(data_capped)

# Stats
all_vals = []
nilai_95_97 = 0
for m in mapel:
    vals = [int(row[m]) for row in data_capped]
    all_vals.extend(vals)
    for v in vals:
        if 95 <= v <= 97:
            nilai_95_97 += 1

max_nilai = max(all_vals)
min_nilai = min(all_vals)
mean = sum(all_vals) / len(all_vals)

print(f"\n📊 FINAL STATISTICS:")
print(f"   Mean: {mean:.2f}")
print(f"   Min: {min_nilai}")
print(f"   Max: {max_nilai} (no more 100!) ✅")
print(f"   Nilai 95-97 (excellent): {nilai_95_97} ({nilai_95_97/len(all_vals)*100:.1f}%)")

print("\n✅ Dataset polished successfully!")
print("   ✓ No perfect 100 scores (more realistic)")
print("   ✓ Max nilai capped at 95-97")
print("   ✓ Natural variation preserved")
print()
