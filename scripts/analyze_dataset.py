#!/usr/bin/env python3
import csv

# Read CSV
with open('dataset_unu.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

mapel = ['mtk', 'inggris', 'agama', 'fisika', 'kimia', 'biologi', 'ekonomi']

print('='*70)
print('ANALISIS DISTRIBUSI NILAI DATASET UNU-MATCH')
print('='*70)

for m in mapel:
    values = [int(row[m]) for row in data]
    mean = sum(values) / len(values)
    values_sorted = sorted(values)
    median = values_sorted[len(values)//2]
    
    below_70 = sum(1 for v in values if v < 70)
    between_70_85 = sum(1 for v in values if 70 <= v < 85)
    above_85 = sum(1 for v in values if v >= 85)
    
    print(f'\n{m.upper()}:')
    print(f'  Mean: {mean:.2f}')
    print(f'  Median: {median}')
    print(f'  Min: {min(values)}')
    print(f'  Max: {max(values)}')
    print(f'  < 70 (Di bawah KKM): {below_70} ({below_70/len(values)*100:.1f}%)')
    print(f'  70-84: {between_70_85} ({between_70_85/len(values)*100:.1f}%)')
    print(f'  >= 85: {above_85} ({above_85/len(values)*100:.1f}%)')

# Overall
print('\n' + '='*70)
print('OVERALL STATISTICS')
print('='*70)

all_values = []
for row in data:
    for m in mapel:
        all_values.append(int(row[m]))

mean_all = sum(all_values) / len(all_values)
below_70_all = sum(1 for v in all_values if v < 70)
above_70_all = sum(1 for v in all_values if v >= 70)

print(f'Total data points: {len(all_values)} (1000 siswa × 7 mapel)')
print(f'Mean semua nilai: {mean_all:.2f}')
print(f'Nilai di bawah KKM 70: {below_70_all} ({below_70_all/len(all_values)*100:.1f}%)')
print(f'Nilai >= KKM 70: {above_70_all} ({above_70_all/len(all_values)*100:.1f}%)')

# Distribusi Prodi
print('\n' + '='*70)
print('DISTRIBUSI PROGRAM STUDI')
print('='*70)
prodi_count = {}
for row in data:
    prodi = row['prodi']
    prodi_count[prodi] = prodi_count.get(prodi, 0) + 1

for prodi, count in sorted(prodi_count.items(), key=lambda x: x[1], reverse=True):
    print(f'{prodi}: {count} students ({count/len(data)*100:.1f}%)')
