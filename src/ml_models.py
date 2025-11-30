"""
NBA ML Models Training Pipeline

This module implements machine learning models for NBA over/under predictions:
1. Logistic Regression (interpretable baseline)
2. Random Forest (feature interactions)
3. XGBoost (high performance gradient boosting)

Includes comprehensive evaluation, hyperparameter tuning, and interpretability analysis.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, log_loss, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import joblib

# Install XGBoost if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

def load_and_prepare_data():
    """Load engineered features and prepare for ML training."""

    print("=== NBA ML MODELS TRAINING ===")
    print("Loading and preparing data...")

    # Load engineered features
    data = pd.read_csv('../data/processed/engineered_features.csv')

    # Convert gameDate to datetime and sort
    data['gameDate'] = pd.to_datetime(data['gameDate'], errors='coerce')
    data = data.sort_values(['fullName', 'gameDate']).reset_index(drop=True)

    print(f"✓ Loaded {len(data):,} games with {data.shape[1]} features")
    print(f"✓ Target variable 'over_threshold' distribution:")
    print(f"  - Over (1): {data['over_threshold'].sum():,} games ({data['over_threshold'].mean():.1%})")
    print(f"  - Under (0): {(data['over_threshold'] == 0).sum():,} games ({(data['over_threshold'] == 0).mean():.1%})")

    return data

def create_temporal_train_test_split(data, test_size=0.2):
    """Create temporal train/test split to prevent look-ahead bias."""

    print(f"\nCreating temporal train/test split ({int((1-test_size)*100)}/{int(test_size*100)})...")

    # Sort by date to ensure temporal order
    data_sorted = data.sort_values('gameDate').copy()

    # Split based on date cutoff
    cutoff_idx = int(len(data_sorted) * (1 - test_size))

    train_data = data_sorted.iloc[:cutoff_idx].copy()
    test_data = data_sorted.iloc[cutoff_idx:].copy()

    print(f"✓ Train: {len(train_data):,} games ({train_data['gameDate'].min().date()} to {train_data['gameDate'].max().date()})")
    print(f"✓ Test: {len(test_data):,} games ({test_data['gameDate'].min().date()} to {test_data['gameDate'].max().date()})")
    print(f"✓ Train over rate: {train_data['over_threshold'].mean():.1%}")
    print(f"✓ Test over rate: {test_data['over_threshold'].mean():.1%}")

    return train_data, test_data

def prepare_features(train_data, test_data):
    """Prepare features for ML models."""

    print("\nPreparing features for ML models...")

    # Define feature categories to exclude
    exclude_features = [
        'gameDate', 'fullName', 'firstName', 'lastName',
        'over_threshold', 'player_threshold', 'points',  # Target and leakage features
        'gameId', 'personId'  # ID features
    ]

    # Get feature columns
    feature_cols = [col for col in train_data.columns if col not in exclude_features]

    # Handle categorical features
    categorical_features = []
    for col in feature_cols:
        if train_data[col].dtype == 'object' or train_data[col].dtype.name == 'category':
            categorical_features.append(col)

    print(f"✓ Found {len(categorical_features)} categorical features: {categorical_features}")

    # Prepare training features
    X_train = train_data[feature_cols].copy()
    y_train = train_data['over_threshold'].copy()

    # Prepare test features
    X_test = test_data[feature_cols].copy()
    y_test = test_data['over_threshold'].copy()

    # Handle categorical features with label encoding
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()

        # Fit on combined data to handle unseen categories
        combined_values = pd.concat([X_train[col], X_test[col]]).astype(str)
        le.fit(combined_values)

        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

        label_encoders[col] = le

    # Handle missing values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    print(f"✓ Prepared {len(feature_cols)} features")
    print(f"✓ Training shape: {X_train.shape}")
    print(f"✓ Test shape: {X_test.shape}")
    print(f"✓ Missing values: Train={X_train.isnull().sum().sum()}, Test={X_test.isnull().sum().sum()}")

    return X_train, X_test, y_train, y_test, feature_cols, label_encoders

def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation."""

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'log_loss': log_loss(y_test, y_proba)
    }

    return metrics, y_pred, y_proba


class ScaledLogisticRegression:
    """Module-level wrapper that scales inputs before calling a sklearn LogisticRegression.

    Defining this at module scope makes it picklable by joblib/pickle.
    """
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))

class MLModelTrainer:
    """Main class for training and evaluating ML models."""

    def __init__(self):
        self.models = {}
        self.results = []
        self.feature_importance = {}

    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train and evaluate Logistic Regression."""

        print("\n--- LOGISTIC REGRESSION ---")

        # Scale features for logistic regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        lr = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'  # Handle class imbalance
        )

        lr.fit(X_train_scaled, y_train)

        # Create scaled model wrapper for consistent interface (module-level class)
        scaled_lr = ScaledLogisticRegression(lr, scaler)

        # Evaluate
        metrics, y_pred, y_proba = evaluate_model(scaled_lr, X_test, y_test, 'Logistic Regression')

        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.3f}")

        # Store results
        self.models['logistic_regression'] = scaled_lr
        self.results.append(metrics)

        # Feature importance (coefficients)
        feature_names = X_train.columns
        importance = np.abs(lr.coef_[0])
        self.feature_importance['logistic_regression'] = dict(zip(feature_names, importance))

        return scaled_lr, metrics

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train and evaluate Random Forest."""

        print("\n--- RANDOM FOREST ---")

        # Train model
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )

        rf.fit(X_train, y_train)

        # Evaluate
        metrics, y_pred, y_proba = evaluate_model(rf, X_test, y_test, 'Random Forest')

        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.3f}")

        # Store results
        self.models['random_forest'] = rf
        self.results.append(metrics)

        # Feature importance
        feature_names = X_train.columns
        importance = rf.feature_importances_
        self.feature_importance['random_forest'] = dict(zip(feature_names, importance))

        return rf, metrics

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train and evaluate XGBoost (if available)."""

        if not XGBOOST_AVAILABLE:
            print("\n--- XGBOOST (SKIPPED) ---")
            print("  XGBoost not available")
            return None, None

        print("\n--- XGBOOST ---")

        # Calculate scale_pos_weight for class balance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        # Train model
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )

        xgb_model.fit(X_train, y_train)

        # Evaluate
        metrics, y_pred, y_proba = evaluate_model(xgb_model, X_test, y_test, 'XGBoost')

        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.3f}")

        # Store results
        self.models['xgboost'] = xgb_model
        self.results.append(metrics)

        # Feature importance
        feature_names = X_train.columns
        importance = xgb_model.feature_importances_
        self.feature_importance['xgboost'] = dict(zip(feature_names, importance))

        return xgb_model, metrics

def analyze_feature_importance(trainer, feature_cols, top_n=20):
    """Analyze and display feature importance across models."""

    print(f"\n=== TOP {top_n} MOST IMPORTANT FEATURES ===")

    for model_name, importance_dict in trainer.feature_importance.items():
        print(f"\n{model_name.upper().replace('_', ' ')}:")
        print("-" * 40)

        # Sort features by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        for i, (feature, importance) in enumerate(sorted_features[:top_n], 1):
            print(f"{i:2d}. {feature:30s} {importance:.4f}")

def save_models_and_results(trainer, feature_cols, label_encoders):
    """Save trained models and results."""

    print("\nSaving models and results...")

    # Create model directory
    model_dir = '../models'
    os.makedirs(model_dir, exist_ok=True)

    # Save models
    for model_name, model in trainer.models.items():
        model_file = f'{model_dir}/{model_name}_model.pkl'
        joblib.dump(model, model_file)
        print(f"✓ Saved {model_name} model: {model_file}")

    # Save label encoders
    encoders_file = f'{model_dir}/label_encoders.pkl'
    joblib.dump(label_encoders, encoders_file)

    # Save feature list
    feature_file = f'{model_dir}/feature_columns.pkl'
    joblib.dump(feature_cols, feature_file)

    # Save results
    results_df = pd.DataFrame(trainer.results)
    results_file = '../data/processed/ml_model_results.csv'
    results_df.to_csv(results_file, index=False)

    # Save feature importance
    importance_data = []
    for model_name, importance_dict in trainer.feature_importance.items():
        for feature, importance in importance_dict.items():
            importance_data.append({
                'model': model_name,
                'feature': feature,
                'importance': importance
            })

    importance_df = pd.DataFrame(importance_data)
    importance_file = '../data/processed/feature_importance.csv'
    importance_df.to_csv(importance_file, index=False)

    print(f"✓ Saved results: {results_file}")
    print(f"✓ Saved feature importance: {importance_file}")

    return results_df

def main():
    """Main ML training workflow."""

    # Step 1: Load and prepare data
    data = load_and_prepare_data()

    # Step 2: Create temporal splits
    train_data, test_data = create_temporal_train_test_split(data)

    # Step 3: Prepare features
    X_train, X_test, y_train, y_test, feature_cols, label_encoders = prepare_features(train_data, test_data)

    # Step 4: Initialize trainer
    trainer = MLModelTrainer()

    # Step 5: Train models
    trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
    trainer.train_random_forest(X_train, y_train, X_test, y_test)
    trainer.train_xgboost(X_train, y_train, X_test, y_test)

    # Step 6: Compare results
    print("\n=== MODEL COMPARISON ===")
    results_df = pd.DataFrame(trainer.results)
    print(results_df[['model', 'accuracy', 'f1_score', 'roc_auc']].to_string(index=False))

    # Step 7: Feature importance analysis
    analyze_feature_importance(trainer, feature_cols)

    # Step 8: Save everything
    final_results = save_models_and_results(trainer, feature_cols, label_encoders)

    # Step 9: Summary
    best_model = results_df.loc[results_df['roc_auc'].idxmax()]

    print("\n=== PHASE 3 COMPLETE ===")
    print(f"✅ Trained {len(trainer.models)} ML models")
    print(f"✅ Best model: {best_model['model']}")
    print(f"✅ Best accuracy: {best_model['accuracy']:.3f}")
    print(f"✅ Best ROC AUC: {best_model['roc_auc']:.3f}")
    print(f"✅ Baseline improvement: +{best_model['accuracy'] - 0.639:.3f} over rolling average")
    print(f"✅ Models saved and ready for deployment")

    return trainer, final_results

if __name__ == "__main__":
    main()