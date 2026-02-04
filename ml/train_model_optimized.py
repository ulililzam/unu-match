#!/usr/bin/env python3
"""
UNU-Match OPTIMIZED Training Script
Focus: Realistic 85%+ accuracy dengan proper validation

Strategy:
1. Balanced feature engineering (not too complex to avoid overfitting)
2. Proper train/test split (20%)
3. Optimized Random Forest + Gradient Boosting Voting
4. No excessive augmentation
5. Cross-validation for reliable metrics
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    f1_score,
    precision_recall_fscore_support
)
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')


class OptimizedUNUMatchTrainer:
    """Optimized trainer for realistic high accuracy"""
    
    def __init__(self, dataset_path='../dataset_unu.csv'):
        self.dataset_path = dataset_path
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_names = None
        self.metrics = {}
        
    def load_data(self):
        """Load dataset"""
        print("📊 Loading dataset...")
        self.df = pd.read_csv(self.dataset_path)
        print(f"✅ Loaded {len(self.df)} records with {self.df['prodi'].nunique()} classes")
        return self
    
    def create_optimized_features(self):
        """Balanced feature engineering - powerful but not overly complex"""
        print("\n🔧 Creating optimized features...")
        
        original = [
            'mtk', 'inggris', 'agama', 'fisika', 'kimia', 
            'biologi', 'ekonomi', 'minat_teknik', 'minat_kesehatan', 
            'minat_bisnis', 'minat_pendidikan', 'hafalan'
        ]
        
        df_features = self.df[original].copy()
        
        # === Core Statistical Features ===
        subjects = ['mtk', 'inggris', 'agama', 'fisika', 'kimia', 'biologi', 'ekonomi']
        df_features['nilai_avg'] = self.df[subjects].mean(axis=1)
        df_features['nilai_std'] = self.df[subjects].std(axis=1)
        df_features['nilai_max'] = self.df[subjects].max(axis=1)
        
        # === Subject Group Averages ===
        df_features['sains_avg'] = self.df[['fisika', 'kimia', 'biologi']].mean(axis=1)
        df_features['exact_avg'] = self.df[['mtk', 'fisika']].mean(axis=1)
        df_features['sosial_avg'] = self.df[['ekonomi', 'agama', 'inggris']].mean(axis=1)
        
        # === Interest Features ===
        interests = ['minat_teknik', 'minat_kesehatan', 'minat_bisnis', 'minat_pendidikan']
        df_features['minat_total'] = self.df[interests].sum(axis=1)
        df_features['minat_max'] = self.df[interests].max(axis=1)
        
        # Dominant interest indicator
        dominant_idx = self.df[interests].values.argmax(axis=1)
        df_features['minat_dominant'] = dominant_idx
        
        # === KEY Interaction Features (Most Important!) ===
        # These capture the relationship between subjects and interests
        df_features['mtk_x_teknik'] = self.df['mtk'] * self.df['minat_teknik']
        df_features['fisika_x_teknik'] = self.df['fisika'] * self.df['minat_teknik']
        df_features['biologi_x_kesehatan'] = self.df['biologi'] * self.df['minat_kesehatan']
        df_features['kimia_x_kesehatan'] = self.df['kimia'] * self.df['minat_kesehatan']
        df_features['ekonomi_x_bisnis'] = self.df['ekonomi'] * self.df['minat_bisnis']
        df_features['agama_x_pendidikan'] = self.df['agama'] * self.df['minat_pendidikan']
        
        # Group x Interest
        df_features['sains_x_kesehatan'] = df_features['sains_avg'] * self.df['minat_kesehatan']
        df_features['exact_x_teknik'] = df_features['exact_avg'] * self.df['minat_teknik']
        df_features['sosial_x_bisnis'] = df_features['sosial_avg'] * self.df['minat_bisnis']
        
        # === Ratio Features ===
        df_features['exact_vs_sosial'] = (df_features['exact_avg'] + 1) / (df_features['sosial_avg'] + 1)
        df_features['sains_vs_ekonomi'] = (df_features['sains_avg'] + 1) / (self.df['ekonomi'] + 1)
        
        # === Binary Indicators ===
        df_features['is_high_math'] = (self.df['mtk'] >= 85).astype(int)
        df_features['is_high_science'] = (df_features['sains_avg'] >= 80).astype(int)
        df_features['is_high_overall'] = (df_features['nilai_avg'] >= 80).astype(int)
        
        # === Weighted Combination Scores ===
        df_features['teknik_score'] = (df_features['exact_avg'] * 0.5 + self.df['minat_teknik'] * 10)
        df_features['kesehatan_score'] = (df_features['sains_avg'] * 0.5 + self.df['minat_kesehatan'] * 10)
        df_features['bisnis_score'] = (df_features['sosial_avg'] * 0.5 + self.df['minat_bisnis'] * 10)
        df_features['pendidikan_score'] = (df_features['nilai_avg'] * 0.4 + self.df['minat_pendidikan'] * 10)
        
        self.X = df_features.values
        self.feature_names = list(df_features.columns)
        self.y = self.label_encoder.fit_transform(self.df['prodi'])
        self.class_names = self.label_encoder.classes_.tolist()
        
        print(f"✅ Created {len(self.feature_names)} features (from {len(original)} original)")
        print(f"✅ Feature matrix: {self.X.shape}")
        
        return self
    
    def split_data(self, test_size=0.2, random_state=42):
        """Standard 80/20 split"""
        print(f"\n✂️  Splitting data (80% train, 20% test)...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.y
        )
        
        print(f"✅ Train: {len(self.X_train)} | Test: {len(self.X_test)}")
        
        return self
    
    def train_optimized_ensemble(self):
        """Train carefully tuned voting ensemble"""
        print("\n🌲 Training Optimized Voting Ensemble...")
        print("   Models: Random Forest + Gradient Boosting")
        
        # Random Forest - Optimized for this dataset
        rf = RandomForestClassifier(
            n_estimators=600,          # More trees = more stable
            max_depth=30,              # Deep enough to capture patterns
            min_samples_split=3,       # Prevent overfitting
            min_samples_leaf=1,        # Allow fine splits
            max_features='sqrt',       # Good default
            class_weight='balanced',   # Handle class imbalance
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting - Complementary algorithm
        gb = GradientBoostingClassifier(
            n_estimators=500,          # More boosting rounds
            max_depth=8,               # Shallow trees for boosting
            learning_rate=0.05,        # Slower learning = better accuracy
            subsample=0.8,             # Bootstrap sampling
            min_samples_split=5,       # Conservative splits
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42
        )
        
        # Voting Classifier with soft voting (use probabilities)
        self.model = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            voting='soft',
            weights=[1.3, 1.0],  # Slightly favor RF
            n_jobs=-1
        )
        
        print("\n⏳ Training ensemble (this will take 3-5 minutes)...")
        self.model.fit(self.X_train, self.y_train)
        
        # Show OOB score from Random Forest
        oob_score = self.model.named_estimators_['rf'].oob_score_
        print(f"✅ Trained! RF OOB Score: {oob_score:.4f} ({oob_score*100:.2f}%)\n")
        
        return self
    
    def evaluate_comprehensive(self):
        """Thorough evaluation"""
        print("\n📈 Evaluating model performance...")
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Metrics
        train_acc = accuracy_score(self.y_train, y_train_pred)
        test_acc = accuracy_score(self.y_test, y_test_pred)
        train_f1 = f1_score(self.y_train, y_train_pred, average='weighted')
        test_f1 = f1_score(self.y_test, y_test_pred, average='weighted')
        
        # Stratified K-Fold CV (gold standard for small datasets)
        print("🔄 Running 10-fold Stratified Cross-Validation...")
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        # CV on full dataset for most reliable estimate
        cv_scores_full = cross_val_score(
            self.model, self.X, self.y, 
            cv=skf, scoring='f1_weighted', n_jobs=-1
        )
        
        # Also CV on training set
        cv_scores_train = cross_val_score(
            self.model, self.X_train, self.y_train, 
            cv=5, scoring='f1_weighted', n_jobs=-1
        )
        
        self.metrics = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'train_f1': float(train_f1),
            'test_f1': float(test_f1),
            'cv_full_mean': float(cv_scores_full.mean()),
            'cv_full_std': float(cv_scores_full.std()),
            'cv_train_mean': float(cv_scores_train.mean()),
            'cv_train_std': float(cv_scores_train.std()),
            'overfitting_gap': float(train_acc - test_acc)
        }
        
        # Print results
        print("\n" + "="*75)
        print("📊 MODEL PERFORMANCE METRICS")
        print("="*75)
        print(f"Training Accuracy:      {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Test Accuracy:          {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Training F1:            {train_f1:.4f}")
        print(f"Test F1:                {test_f1:.4f}")
        print("-"*75)
        print(f"CV (Full Dataset):      {cv_scores_full.mean():.4f} ± {cv_scores_full.std():.4f}")
        print(f"                        ({cv_scores_full.mean()*100:.2f}% ± {cv_scores_full.std()*100:.2f}%)")
        print(f"CV (Train Set):         {cv_scores_train.mean():.4f} ± {cv_scores_train.std():.4f}")
        print(f"                        ({cv_scores_train.mean()*100:.2f}% ± {cv_scores_train.std()*100:.2f}%)")
        print("-"*75)
        print(f"Overfitting Gap:        {(train_acc - test_acc)*100:.2f}%")
        print("="*75 + "\n")
        
        # Classification report
        print("📋 Detailed Classification Report:")
        print(classification_report(
            self.y_test, y_test_pred, 
            target_names=self.class_names,
            digits=3
        ))
        
        # Per-class analysis
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_test, y_test_pred, average=None
        )
        
        print("🎯 Per-Class Performance (sorted by F1):")
        print("="*75)
        class_metrics = []
        for i, name in enumerate(self.class_names):
            class_metrics.append((name, precision[i], recall[i], f1[i], support[i]))
        class_metrics.sort(key=lambda x: x[3], reverse=True)
        
        for name, p, r, f, s in class_metrics:
            print(f"{name:40s} P:{p:.3f} R:{r:.3f} F1:{f:.3f} (n={int(s):2d})")
        print("="*75 + "\n")
        
        return self
    
    def analyze_features(self):
        """Feature importance from Random Forest"""
        print("\n🔍 Feature Importance Analysis...")
        
        rf_model = self.model.named_estimators_['rf']
        importances = rf_model.feature_importances_
        
        feat_imp_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = feat_imp_df
        
        print("\n📊 TOP 20 MOST IMPORTANT FEATURES")
        print("="*75)
        for idx, row in feat_imp_df.head(20).iterrows():
            bar_len = int(row['importance'] * 100)
            bar = "█" * bar_len
            print(f"{row['feature']:30s} {bar} {row['importance']:.4f}")
        print("="*75 + "\n")
        
        return self
    
    def export_models(self, output_dir='../models'):
        """Export all models"""
        print(f"\n💾 Exporting models to {output_dir}...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Python models
        joblib.dump(self.model, output_path / 'voting_ensemble.pkl')
        joblib.dump(self.label_encoder, output_path / 'label_encoder.pkl')
        
        # JavaScript Random Forest
        rf_model = self.model.named_estimators_['rf']
        trees_data = []
        n_trees_export = min(100, len(rf_model.estimators_))
        
        for tree in rf_model.estimators_[:n_trees_export]:
            tree_structure = self._export_tree(tree.tree_)
            trees_data.append(tree_structure)
        
        model_json = {
            'model_type': 'RandomForest',
            'n_estimators': len(trees_data),
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'trees': trees_data
        }
        
        with open(output_path / 'rf_model.json', 'w') as f:
            json.dump(model_json, f, indent=2)
        
        # Feature importance
        with open(output_path / 'feature_importance.json', 'w') as f:
            json.dump({
                'features': self.feature_importance.to_dict('records')
            }, f, indent=2)
        
        # Metadata
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'model_type': 'Voting Ensemble (RF + GB)',
            'dataset_size': len(self.df),
            'n_features': len(self.feature_names),
            'n_classes': len(self.class_names),
            'metrics': self.metrics,
            'hyperparameters': {
                'rf_n_estimators': rf_model.n_estimators,
                'rf_max_depth': rf_model.max_depth,
                'gb_n_estimators': self.model.named_estimators_['gb'].n_estimators,
                'gb_learning_rate': self.model.named_estimators_['gb'].learning_rate
            }
        }
        
        with open(output_path / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Export complete!")
        print(f"   - voting_ensemble.pkl (Python)")
        print(f"   - rf_model.json (JavaScript, {n_trees_export} trees)")
        print(f"   - feature_importance.json")
        print(f"   - model_metadata.json")
        
        return self
    
    def _export_tree(self, tree):
        """Export tree structure"""
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
    
    def run_full_pipeline(self):
        """Execute complete training pipeline"""
        print("\n" + "="*75)
        print("🚀 UNU-MATCH OPTIMIZED TRAINING PIPELINE")
        print("   Goal: Realistic 85%+ accuracy with proper validation")
        print("="*75 + "\n")
        
        (self.load_data()
             .create_optimized_features()
             .split_data()
             .train_optimized_ensemble()
             .evaluate_comprehensive()
             .analyze_features()
             .export_models())
        
        print("\n" + "="*75)
        print("✅ TRAINING COMPLETE!")
        print("="*75)
        
        # Final summary
        test_acc = self.metrics['test_accuracy'] * 100
        cv_acc = self.metrics['cv_full_mean'] * 100
        cv_std = self.metrics['cv_full_std'] * 100
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"   Test Accuracy:       {test_acc:.2f}%")
        print(f"   CV Accuracy:         {cv_acc:.2f}% ± {cv_std:.2f}%")
        print(f"   Improvement:         +{test_acc - 70:.2f}% from baseline (70%)")
        
        if test_acc >= 85:
            print(f"\n🎉🎉🎉 SUCCESS! Target 85%+ achieved! 🎉🎉🎉")
        elif cv_acc >= 85:
            print(f"\n✨ CV shows {cv_acc:.2f}% - Excellent performance!")
        else:
            print(f"\n📈 Solid improvement! Gap to 85%: {85 - test_acc:.2f}%")
        
        print(f"\n💡 The ensemble is ready for deployment!")
        
        return self


def main():
    """Main execution"""
    trainer = OptimizedUNUMatchTrainer(dataset_path='../dataset_unu.csv')
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()
