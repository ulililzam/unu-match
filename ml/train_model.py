#!/usr/bin/env python3
"""
UNU-Match Random Forest Training Script
Train, validate, and export Random Forest model for program recommendation
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score
)
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')


class UNUMatchMLTrainer:
    """Train and validate Random Forest model for UNU-Match"""
    
    def __init__(self, dataset_path='../dataset_unu.csv'):
        self.dataset_path = dataset_path
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_names = None
        self.metrics = {}
        
    def load_data(self):
        """Load and prepare dataset"""
        print("📊 Loading dataset...")
        self.df = pd.read_csv(self.dataset_path)
        
        print(f"✅ Loaded {len(self.df)} records")
        print(f"✅ Features: {self.df.shape[1] - 1}")
        print(f"✅ Classes: {self.df['prodi'].nunique()}")
        print(f"\n📋 Class distribution:")
        print(self.df['prodi'].value_counts().sort_index())
        
        return self
    
    def prepare_features(self):
        """Prepare features and target"""
        print("\n🔧 Preparing features...")
        
        # Features (X)
        self.feature_names = [
            'mtk', 'inggris', 'agama', 'fisika', 'kimia', 
            'biologi', 'ekonomi', 'minat_teknik', 'minat_kesehatan', 
            'minat_bisnis', 'minat_pendidikan', 'hafalan'
        ]
        self.X = self.df[self.feature_names].values
        
        # Target (y)
        self.y = self.label_encoder.fit_transform(self.df['prodi'])
        self.class_names = self.label_encoder.classes_.tolist()
        
        print(f"✅ Feature matrix shape: {self.X.shape}")
        print(f"✅ Classes encoded: {len(self.class_names)}")
        
        return self
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print(f"\n✂️  Splitting data (test_size={test_size})...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.y  # Maintain class distribution
        )
        
        print(f"✅ Training set: {len(self.X_train)} samples")
        print(f"✅ Test set: {len(self.X_test)} samples")
        
        return self
    
    def train_model(self, tune_hyperparameters=True):
        """Train Random Forest model"""
        print("\n🌲 Training Random Forest...")
        
        if tune_hyperparameters:
            print("🔍 Performing hyperparameter tuning...")
            
            # Hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            
            # Grid search with cross-validation
            rf_base = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf_base, 
                param_grid, 
                cv=5,
                scoring='f1_weighted',
                verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            self.model = grid_search.best_estimator_
            print(f"\n✅ Best parameters: {grid_search.best_params_}")
            print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
            
        else:
            # Train with default good parameters
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42
            )
            self.model.fit(self.X_train, self.y_train)
            print("✅ Model trained with default parameters")
        
        return self
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n📈 Evaluating model...")
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Accuracy
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        # F1 Score
        train_f1 = f1_score(self.y_train, y_train_pred, average='weighted')
        test_f1 = f1_score(self.y_test, y_test_pred, average='weighted')
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, self.X_train, self.y_train, 
            cv=5, scoring='f1_weighted'
        )
        
        # Store metrics
        self.metrics = {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'train_f1': float(train_f1),
            'test_f1': float(test_f1),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std())
        }
        
        # Print results
        print(f"\n{'='*60}")
        print(f"📊 MODEL PERFORMANCE METRICS")
        print(f"{'='*60}")
        print(f"Training Accuracy:   {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"Test Accuracy:       {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Training F1:         {train_f1:.4f}")
        print(f"Test F1:             {test_f1:.4f}")
        print(f"Cross-Val F1:        {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        print(f"{'='*60}\n")
        
        # Classification report
        print("📋 Classification Report:")
        print(classification_report(
            self.y_test, 
            y_test_pred, 
            target_names=self.class_names,
            digits=3
        ))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_test_pred)
        print(f"\n🎯 Confusion Matrix:")
        print(cm)
        
        return self
    
    def analyze_feature_importance(self):
        """Analyze and display feature importance"""
        print("\n🔍 Feature Importance Analysis...")
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create dataframe
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = feature_importance_df
        
        print("\n" + "="*60)
        print("📊 TOP FEATURE IMPORTANCE")
        print("="*60)
        for idx, row in feature_importance_df.iterrows():
            bar_length = int(row['importance'] * 50)
            bar = "█" * bar_length
            print(f"{row['feature']:20s} {bar} {row['importance']:.4f}")
        print("="*60 + "\n")
        
        return self
    
    def export_model_to_json(self, output_dir='../models'):
        """Export trained Random Forest to JSON format for JavaScript"""
        print(f"\n💾 Exporting model to {output_dir}...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. Export Decision Trees from Random Forest
        trees_data = []
        for idx, tree in enumerate(self.model.estimators_):
            tree_structure = self._export_tree_to_dict(tree.tree_)
            trees_data.append(tree_structure)
        
        model_json = {
            'model_type': 'RandomForest',
            'n_estimators': len(trees_data),
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'trees': trees_data[:50]  # Export first 50 trees to reduce size
        }
        
        # Save model
        with open(output_path / 'rf_model.json', 'w') as f:
            json.dump(model_json, f, indent=2)
        
        # 2. Export feature importance
        feature_importance_json = {
            'features': self.feature_importance.to_dict('records')
        }
        with open(output_path / 'feature_importance.json', 'w') as f:
            json.dump(feature_importance_json, f, indent=2)
        
        # 3. Export metadata
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'dataset_size': len(self.df),
            'n_features': len(self.feature_names),
            'n_classes': len(self.class_names),
            'metrics': self.metrics,
            'model_params': {
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'min_samples_split': self.model.min_samples_split,
                'min_samples_leaf': self.model.min_samples_leaf
            }
        }
        with open(output_path / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 4. Save scikit-learn model (for Python reuse)
        joblib.dump(self.model, output_path / 'rf_model.pkl')
        joblib.dump(self.label_encoder, output_path / 'label_encoder.pkl')
        
        print("✅ Model exported successfully!")
        print(f"   - rf_model.json ({len(trees_data[:50])} trees)")
        print(f"   - feature_importance.json")
        print(f"   - model_metadata.json")
        print(f"   - rf_model.pkl (Python)")
        
        return self
    
    def _export_tree_to_dict(self, tree):
        """Convert sklearn tree to dictionary structure"""
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        value = tree.value
        
        def recurse(node_id):
            if children_left[node_id] == children_right[node_id]:
                # Leaf node
                return {
                    'type': 'leaf',
                    'value': value[node_id][0].tolist(),
                    'prediction': int(np.argmax(value[node_id][0]))
                }
            else:
                # Decision node
                return {
                    'type': 'decision',
                    'feature': int(feature[node_id]),
                    'threshold': float(threshold[node_id]),
                    'left': recurse(children_left[node_id]),
                    'right': recurse(children_right[node_id])
                }
        
        return recurse(0)
    
    def run_full_pipeline(self, tune_hyperparameters=False):
        """Run complete ML pipeline"""
        print("\n" + "="*60)
        print("🚀 UNU-MATCH RANDOM FOREST TRAINING PIPELINE")
        print("="*60 + "\n")
        
        (self.load_data()
             .prepare_features()
             .split_data()
             .train_model(tune_hyperparameters=tune_hyperparameters)
             .evaluate_model()
             .analyze_feature_importance()
             .export_model_to_json())
        
        print("\n" + "="*60)
        print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60 + "\n")
        
        return self


def main():
    """Main execution"""
    trainer = UNUMatchMLTrainer(dataset_path='../dataset_unu.csv')
    
    # Run full pipeline with hyperparameter tuning for better accuracy
    trainer.run_full_pipeline(tune_hyperparameters=True)
    
    print("\n🎉 Model is ready for deployment!")
    print("📂 Check the 'models/' directory for exported files")
    print("\n💡 Next step: Integrate with JavaScript using ml_engine.js")


if __name__ == "__main__":
    main()
