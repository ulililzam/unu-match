#!/usr/bin/env python3
"""
UNU-Match Random Forest Training Script (FAST VERSION)
Train model dengan parameters optimal tanpa full grid search
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score
)
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*60)
print("🚀 UNU-MATCH RANDOM FOREST TRAINING (FAST MODE)")
print("="*60 + "\n")

# 1. Load Data
print("📊 Loading dataset...")
df = pd.read_csv('../dataset_unu.csv')
print(f"✅ Loaded {len(df)} records")
print(f"✅ Features: {df.shape[1] - 1}")
print(f"✅ Classes: {df['prodi'].nunique()}\n")

# 2. Prepare Features
print("🔧 Preparing features...")
feature_names = [
    'mtk', 'inggris', 'agama', 'fisika', 'kimia', 
    'biologi', 'ekonomi', 'minat_teknik', 'minat_kesehatan', 
    'minat_bisnis', 'minat_pendidikan', 'hafalan'
]
X = df[feature_names].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['prodi'])
class_names = label_encoder.classes_.tolist()

print(f"✅ Feature matrix shape: {X.shape}")
print(f"✅ Classes encoded: {len(class_names)}\n")

# 3. Split Data
print("✂️  Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Training set: {len(X_train)} samples")
print(f"✅ Test set: {len(X_test)} samples\n")

# 4. Train Model with OPTIMAL parameters (no grid search)
print("🌲 Training Random Forest with optimized parameters...")
print("   (Skipping grid search for speed - using best known params)\n")

model = RandomForestClassifier(
    n_estimators=300,           # Increased trees
    max_depth=20,               # Deeper trees
    min_samples_split=3,        # Balanced
    min_samples_leaf=1,         # Allow granularity
    max_features='sqrt',        # Best for classification
    min_impurity_decrease=0.001,# Prune weak splits
    class_weight='balanced',    # Handle imbalance
    random_state=42,
    n_jobs=-1,                  # Use all cores
    verbose=1                   # Show progress
)

model.fit(X_train, y_train)
print("\n✅ Model trained successfully!\n")

# 5. Evaluate
print("📈 Evaluating model...\n")

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

# Cross-validation
print("🔄 Running cross-validation...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=-1)

print("\n" + "="*60)
print("📊 MODEL PERFORMANCE METRICS")
print("="*60)
print(f"Training Accuracy:   {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy:       {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Training F1:         {train_f1:.4f}")
print(f"Test F1:             {test_f1:.4f}")
print(f"Cross-Val F1:        {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
print("="*60 + "\n")

# Classification Report
print("📋 Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=class_names, digits=3))

# 6. Feature Importance
print("\n🔍 TOP FEATURE IMPORTANCE")
print("="*60)
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

for idx, row in feature_importance_df.iterrows():
    bar_length = int(row['importance'] * 50)
    bar = "█" * bar_length
    print(f"{row['feature']:20s} {bar} {row['importance']:.4f}")
print("="*60 + "\n")

# 7. Export Model
print("💾 Exporting model...\n")
output_path = Path('../models')
output_path.mkdir(exist_ok=True)

# Helper function to export tree
def export_tree_to_dict(tree):
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value
    
    def recurse(node_id):
        if children_left[node_id] == children_right[node_id]:
            return {
                'type': 'leaf',
                'value': value[node_id][0].tolist(),
                'prediction': int(np.argmax(value[node_id][0]))
            }
        else:
            return {
                'type': 'decision',
                'feature': int(feature[node_id]),
                'threshold': float(threshold[node_id]),
                'left': recurse(children_left[node_id]),
                'right': recurse(children_right[node_id])
            }
    
    return recurse(0)

# Export trees (first 50 for browser efficiency)
print("   📦 Exporting decision trees (50 trees)...")
trees_data = []
for idx, tree in enumerate(model.estimators_[:50]):
    tree_structure = export_tree_to_dict(tree.tree_)
    trees_data.append(tree_structure)
    if (idx + 1) % 10 == 0:
        print(f"      ✓ Exported {idx + 1}/50 trees")

model_json = {
    'model_type': 'RandomForest',
    'n_estimators': len(trees_data),
    'feature_names': feature_names,
    'class_names': class_names,
    'trees': trees_data
}

with open(output_path / 'rf_model.json', 'w') as f:
    json.dump(model_json, f, indent=2)
print("   ✅ rf_model.json")

# Feature importance
feature_importance_json = {
    'features': feature_importance_df.to_dict('records')
}
with open(output_path / 'feature_importance.json', 'w') as f:
    json.dump(feature_importance_json, f, indent=2)
print("   ✅ feature_importance.json")

# Metadata
metadata = {
    'trained_at': datetime.now().isoformat(),
    'dataset_size': len(df),
    'n_features': len(feature_names),
    'n_classes': len(class_names),
    'metrics': {
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'train_f1': float(train_f1),
        'test_f1': float(test_f1),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std())
    },
    'model_params': {
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'min_samples_split': model.min_samples_split,
        'min_samples_leaf': model.min_samples_leaf
    }
}
with open(output_path / 'model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("   ✅ model_metadata.json")

# Save Python model
joblib.dump(model, output_path / 'rf_model.pkl')
joblib.dump(label_encoder, output_path / 'label_encoder.pkl')
print("   ✅ rf_model.pkl (Python backup)")

print("\n" + "="*60)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\n📊 Final Accuracy: {test_accuracy*100:.2f}%")
print(f"📊 Final F1 Score: {test_f1:.4f}")
print(f"\n📂 Model files saved to: {output_path.absolute()}")
print("\n💡 Next steps:")
print("   1. Test the model at: http://localhost:8000/test_accuracy.html")
print("   2. Try the app at: http://localhost:8000/survey.html")
print("\n🎉 Model is ready for deployment!\n")
