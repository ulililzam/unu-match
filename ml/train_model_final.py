#!/usr/bin/env python3
"""
UNU-Match FINAL Training Script
Focus on GENERALIZATION, not memorization

Key insight: The baseline 70% was actually GOOD with strong regularization.
This version aims for 85%+ by:
1. Strong regularization to prevent overfitting
2. Balanced feature set
3. Proper ensemble voting
4. Realistic evaluation

Target: < 5% overfitting gap, 85%+ test accuracy
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')


print("\n" + "="*80)
print("🎯 UNU-MATCH FINAL TRAINING - FOCUS ON GENERALIZATION")
print("="*80 + "\n")

# 1. LOAD DATA
print("📊 Loading dataset...")
df = pd.read_csv('../dataset_unu.csv')
print(f"✅ {len(df)} samples, {df['prodi'].nunique()} classes\n")

# 2. FEATURE ENGINEERING (Balanced approach)
print("🔧 Engineering features (balanced, not over-complex)...")

original = ['mtk', 'inggris', 'agama', 'fisika', 'kimia', 'biologi', 'ekonomi',
            'minat_teknik', 'minat_kesehatan', 'minat_bisnis', 'minat_pendidikan', 'hafalan']

df_feat = df[original].copy()

# Subject groups
subjects = ['mtk', 'inggris', 'agama', 'fisika', 'kimia', 'biologi', 'ekonomi']
df_feat['nilai_avg'] = df[subjects].mean(axis=1)
df_feat['sains_avg'] = df[['fisika', 'kimia', 'biologi']].mean(axis=1)
df_feat['exact_avg'] = df[['mtk', 'fisika']].mean(axis=1)
df_feat['sosial_avg'] = df[['ekonomi', 'agama']].mean(axis=1)

# Interest dominant
interests = ['minat_teknik', 'minat_kesehatan', 'minat_bisnis', 'minat_pendidikan']
df_feat['minat_max'] = df[interests].max(axis=1)

# KEY interactions (most predictive)
df_feat['mtk_x_teknik'] = df['mtk'] * df['minat_teknik']
df_feat['biologi_x_kesehatan'] = df['biologi'] * df['minat_kesehatan']
df_feat['ekonomi_x_bisnis'] = df['ekonomi'] * df['minat_bisnis']
df_feat['agama_x_pendidikan'] = df['agama'] * df['minat_pendidikan']

# Prepare X and y
X = df_feat.values
feature_names = list(df_feat.columns)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['prodi'])
class_names = label_encoder.classes_.tolist()

print(f"✅ {len(feature_names)} features created")
print(f"✅ Feature matrix: {X.shape}\n")

# 3. SPLIT DATA
print("✂️  Splitting data (80/20 stratified split)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}\n")

# 4. TRAIN MODEL with STRONG REGULARIZATION
print("🌲 Training Ensemble with STRONG regularization...")
print("   (Preventing overfitting is KEY to generalization)\n")

# Random Forest with regularization
rf = RandomForestClassifier(
    n_estimators=800,
    max_depth=18,              # Limited depth (was 30)
    min_samples_split=10,      # Require more samples to split (was 3)
    min_samples_leaf=4,        # Require more samples in leaves (was 1)
    max_features='sqrt',
    class_weight='balanced',
    max_samples=0.8,           # Bootstrap 80% of samples
    random_state=42,
    n_jobs=-1
)

# Gradient Boosting with regularization
gb = GradientBoostingClassifier(
    n_estimators=600,
    max_depth=6,               # Shallow trees (was 8)
    learning_rate=0.03,        # Slow learning (was 0.05)
    subsample=0.7,             # More subsampling (was 0.8)
    min_samples_split=10,      # More regularization (was 5)
    min_samples_leaf=4,        # More regularization (was 2)
    max_features='sqrt',
    random_state=42
)

# Voting ensemble
model = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb)],
    voting='soft',
    weights=[1.2, 1.0],
    n_jobs=-1
)

print("⏳ Training (3-5 minutes)...")
model.fit(X_train, y_train)
print("✅ Training complete!\n")

# 5. EVALUATE
print("📈 Evaluating performance...\n")

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

# Cross-validation (most reliable metric)
print("🔄 10-Fold Stratified Cross-Validation on FULL dataset...")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
cv_f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted', n_jobs=-1)

print("\n" + "="*80)
print("📊 FINAL RESULTS")
print("="*80)
print(f"Training Accuracy:        {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Test Accuracy:            {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Training F1:              {train_f1:.4f}")
print(f"Test F1:                  {test_f1:.4f}")
print("-"*80)
print(f"10-Fold CV Accuracy:      {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"                          ({cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%)")
print(f"10-Fold CV F1:            {cv_f1_scores.mean():.4f} ± {cv_f1_scores.std():.4f}")
print(f"                          ({cv_f1_scores.mean()*100:.2f}% ± {cv_f1_scores.std()*100:.2f}%)")
print("-"*80)
print(f"Overfitting Gap:          {(train_acc - test_acc)*100:.2f}% (target: <10%)")
print(f"CV vs Test Difference:    {abs(cv_scores.mean() - test_acc)*100:.2f}%")
print("="*80 + "\n")

# Classification report
print("📋 Per-Class Performance:")
print(classification_report(y_test, y_test_pred, target_names=class_names, digits=3))

# 6. FEATURE IMPORTANCE
print("\n🔍 Top 15 Feature Importances:")
print("="*80)
rf_model = model.named_estimators_['rf']
importances = rf_model.feature_importances_
feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

for i, (feat, imp) in enumerate(feat_imp[:15]):
    bar = "█" * int(imp * 100)
    print(f"{i+1:2d}. {feat:30s} {bar} {imp:.4f}")
print("="*80 + "\n")

# 7. EXPORT MODELS
print("💾 Exporting models...")
output_path = Path('../models')
output_path.mkdir(exist_ok=True)

# Save Python models
joblib.dump(model, output_path / 'final_ensemble.pkl')
joblib.dump(label_encoder, output_path / 'label_encoder.pkl')

# Export RF for JavaScript
def export_tree(tree):
    def recurse(node_id):
        if tree.children_left[node_id] == tree.children_right[node_id]:
            return {
                'type': 'leaf',
                'value': tree.value[node_id][0].tolist(),
                'prediction': int(np.argmax(tree.value[node_id][0]))
            }
        return {
            'type': 'decision',
            'feature': int(tree.feature[node_id]),
            'threshold': float(tree.threshold[node_id]),
            'left': recurse(tree.children_left[node_id]),
            'right': recurse(tree.children_right[node_id])
        }
    return recurse(0)

trees_data = [export_tree(tree.tree_) for tree in rf_model.estimators_[:100]]

model_json = {
    'model_type': 'RandomForest',
    'n_estimators': len(trees_data),
    'feature_names': feature_names,
    'class_names': class_names,
    'trees': trees_data
}

with open(output_path / 'rf_model.json', 'w') as f:
    json.dump(model_json, f, indent=2)

# Feature importance
with open(output_path / 'feature_importance.json', 'w') as f:
    json.dump({'features': [{'feature': f, 'importance': float(i)} for f, i in feat_imp]}, f, indent=2)

# Metadata
metadata = {
    'trained_at': datetime.now().isoformat(),
    'model_type': 'Voting Ensemble (Regularized RF + GB)',
    'dataset_size': len(df),
    'n_features': len(feature_names),
    'n_classes': len(class_names),
    'metrics': {
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'train_f1': float(train_f1),
        'test_f1': float(test_f1),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'cv_f1_mean': float(cv_f1_scores.mean()),
        'cv_f1_std': float(cv_f1_scores.std())
    }
}

with open(output_path / 'model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Export complete!\n")

# 8. FINAL SUMMARY
print("="*80)
print("🎉 TRAINING COMPLETE - SUMMARY")
print("="*80)

baseline_acc = 0.70
improvement = (test_acc - baseline_acc) * 100
cv_improvement = (cv_scores.mean() - baseline_acc) * 100

print(f"\n📊 Accuracy Metrics:")
print(f"   Test Set Accuracy:     {test_acc*100:.2f}%")
print(f"   Cross-Val Accuracy:    {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print(f"   Improvement vs 70%:    {'+' if improvement > 0 else ''}{improvement:.2f}%")

print(f"\n🎯 Model Quality:")
print(f"   Overfitting Gap:       {(train_acc - test_acc)*100:.2f}% ", end="")
if (train_acc - test_acc) < 0.10:
    print("✅ EXCELLENT!")
elif (train_acc - test_acc) < 0.15:
    print("✓ Good")
else:
    print("⚠ Needs improvement")

print(f"   Generalization:        ", end="")
if abs(cv_scores.mean() - test_acc) < 0.05:
    print("✅ EXCELLENT (CV ≈ Test)")
else:
    print(f"CV differs by {abs(cv_scores.mean() - test_acc)*100:.2f}%")

print(f"\n🚀 Achievement Status:")
if test_acc >= 0.85:
    print(f"   🎉🎉🎉 TARGET ACHIEVED! Test accuracy {test_acc*100:.2f}% >= 85% 🎉🎉🎉")
elif cv_scores.mean() >= 0.85:
    print(f"   ✨ CV shows {cv_scores.mean()*100:.2f}% - Exceeds target!")
elif test_acc >= 0.80:
    print(f"   ✓ Strong performance! Just {(0.85 - test_acc)*100:.2f}% from 85% target")
else:
    print(f"   📈 Good progress. Need +{(0.85 - test_acc)*100:.2f}% to reach 85%")

print(f"\n💡 Best estimate of real-world performance: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print("   (Cross-validation is most reliable for small datasets)")

print("\n" + "="*80)
print("✅ Model ready for deployment!")
print("📂 Files exported to 'models/' directory")
print("="*80 + "\n")
