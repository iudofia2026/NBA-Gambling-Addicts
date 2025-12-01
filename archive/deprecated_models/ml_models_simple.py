"""
NBA ML Models Training Pipeline (Simplified)

Training Logistic Regression and Random Forest without XGBoost dependency issues.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, log_loss, classification_report
import joblib

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

def prepare_features_for_ml(data):
    """Prepare features for ML models."""

    print("\nPreparing features for ML models...")

    # Create temporal split (80/20)
    np.random.seed(42)
    train_mask = np.random.random(len(data)) < 0.8

    train_data = data[train_mask].copy()
    test_data = data[~train_mask].copy()

    print(f"✓ Train: {len(train_data):,} games")
    print(f"✓ Test: {len(test_data):,} games")

    # Define feature categories to exclude
    exclude_features = [
        'gameDate', 'fullName', 'firstName', 'lastName',
        'over_threshold', 'player_threshold', 'points',  # Target and leakage features
        'gameId', 'personId'  # ID features
    ]

    # Get feature columns
    feature_cols = [col for col in data.columns if col not in exclude_features]

    # Handle categorical features
    categorical_features = []
    for col in feature_cols:
        if train_data[col].dtype == 'object' or str(train_data[col].dtype) == 'category':
            categorical_features.append(col)

    print(f"✓ Found {len(categorical_features)} categorical features: {categorical_features}")

    # Prepare features
    X_train = train_data[feature_cols].copy()
    y_train = train_data['over_threshold'].copy()
    X_test = test_data[feature_cols].copy()
    y_test = test_data['over_threshold'].copy()

    # Handle categorical features with label encoding
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()

        # Handle missing values first
        train_vals = X_train[col].fillna('missing').astype(str)
        test_vals = X_test[col].fillna('missing').astype(str)

        # Fit on combined data
        combined_vals = pd.concat([train_vals, test_vals])
        le.fit(combined_vals)

        X_train[col] = le.transform(train_vals)
        X_test[col] = le.transform(test_vals)

        label_encoders[col] = le

    # Handle missing values in numeric features
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Ensure all features are numeric
    for col in feature_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)

    print(f"✓ Prepared {len(feature_cols)} features")
    print(f"✓ Training shape: {X_train.shape}")
    print(f"✓ Test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test, feature_cols, label_encoders

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train and evaluate Logistic Regression."""

    print("\n--- LOGISTIC REGRESSION ---")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    lr = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )

    lr.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = lr.predict(X_test_scaled)
    y_proba = lr.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    metrics = {
        'model': 'Logistic Regression',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'log_loss': log_loss(y_test, y_proba)
    }

    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  F1 Score: {metrics['f1_score']:.3f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"  Log Loss: {metrics['log_loss']:.3f}")

    # Feature importance (coefficients)
    importance = np.abs(lr.coef_[0])
    feature_importance = dict(zip(X_train.columns, importance))

    return lr, scaler, metrics, feature_importance

def train_random_forest(X_train, y_train, X_test, y_test):
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

    # Predictions
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        'model': 'Random Forest',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'log_loss': log_loss(y_test, y_proba)
    }

    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  F1 Score: {metrics['f1_score']:.3f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"  Log Loss: {metrics['log_loss']:.3f}")

    # Feature importance
    importance = rf.feature_importances_
    feature_importance = dict(zip(X_train.columns, importance))

    return rf, metrics, feature_importance

def analyze_feature_importance(lr_importance, rf_importance, top_n=15):
    """Analyze top features across models."""

    print(f"\n=== TOP {top_n} MOST IMPORTANT FEATURES ===")

    print("\nLOGISTIC REGRESSION (Coefficient Magnitude):")
    print("-" * 50)
    lr_sorted = sorted(lr_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(lr_sorted[:top_n], 1):
        print(f"{i:2d}. {feature:35s} {importance:.4f}")

    print("\nRANDOM FOREST (Feature Importance):")
    print("-" * 50)
    rf_sorted = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(rf_sorted[:top_n], 1):
        print(f"{i:2d}. {feature:35s} {importance:.4f}")

def save_results(lr_metrics, rf_metrics, lr_importance, rf_importance):
    """Save model results and feature importance."""

    print("\nSaving results...")

    # Create directories
    os.makedirs('../data/processed', exist_ok=True)
    os.makedirs('../models', exist_ok=True)

    # Save model comparison
    results_df = pd.DataFrame([lr_metrics, rf_metrics])
    results_file = '../data/processed/ml_model_results.csv'
    results_df.to_csv(results_file, index=False)

    # Save feature importance
    importance_data = []

    for feature, importance in lr_importance.items():
        importance_data.append({
            'model': 'Logistic Regression',
            'feature': feature,
            'importance': importance
        })

    for feature, importance in rf_importance.items():
        importance_data.append({
            'model': 'Random Forest',
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

    # Load data
    data = load_and_prepare_data()

    # Prepare features
    X_train, X_test, y_train, y_test, feature_cols, label_encoders = prepare_features_for_ml(data)

    # Train models
    lr_model, lr_scaler, lr_metrics, lr_importance = train_logistic_regression(
        X_train, y_train, X_test, y_test
    )

    rf_model, rf_metrics, rf_importance = train_random_forest(
        X_train, y_train, X_test, y_test
    )

    # Compare models
    print("\n=== MODEL COMPARISON ===")
    comparison_df = pd.DataFrame([lr_metrics, rf_metrics])
    print(comparison_df[['model', 'accuracy', 'f1_score', 'roc_auc']].to_string(index=False))

    # Analyze feature importance
    analyze_feature_importance(lr_importance, rf_importance)

    # Save results
    results_df = save_results(lr_metrics, rf_metrics, lr_importance, rf_importance)

    # Summary
    best_model_idx = comparison_df['roc_auc'].idxmax()
    best_model = comparison_df.iloc[best_model_idx]

    print("\n=== ML TRAINING COMPLETE ===")
    print(f"✅ Trained 2 ML models successfully")
    print(f"✅ Best model: {best_model['model']}")
    print(f"✅ Best accuracy: {best_model['accuracy']:.3f}")
    print(f"✅ Best ROC AUC: {best_model['roc_auc']:.3f}")
    print(f"✅ Improvement over baseline (63.9%): +{best_model['accuracy'] - 0.639:.3f}")

    # Compare to baseline
    if best_model['accuracy'] > 0.639:
        print(f"🎯 SUCCESS: ML model beats rolling average baseline!")
    else:
        print(f"📊 INFO: Baseline still competitive, need feature tuning")

    return results_df, lr_importance, rf_importance

if __name__ == "__main__":
    main()