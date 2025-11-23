"""
NBA ML Models - Fixed Data Leakage

Final ML models with proper feature selection to prevent data leakage.
Removes any features that could contain future information about the target.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import joblib

def load_and_prepare_data():
    """Load data and remove any potential data leakage features."""

    print("=== NBA ML MODELS - DATA LEAKAGE FIXED ===")
    print("Loading and preparing clean features...")

    # Load engineered features
    data = pd.read_csv('../data/processed/engineered_features.csv')

    # Convert gameDate and sort
    data['gameDate'] = pd.to_datetime(data['gameDate'], errors='coerce')
    data = data.sort_values(['fullName', 'gameDate']).reset_index(drop=True)

    print(f"✓ Loaded {len(data):,} games with {data.shape[1]} features")

    return data

def remove_data_leakage_features(data):
    """Remove features that could contain information about the current game result."""

    print("\nRemoving data leakage features...")

    # Features that definitely contain future information or direct game results
    leakage_features = [
        'points',  # Direct target leakage
        'points_vs_threshold',  # Direct calculation using points
        'points_vs_rolling_avg',  # Uses current game points
        'points_vs_age_expected',  # Uses current game points
        'points_per_minute',  # Uses current game points
        'points_per_shot',  # Uses current game points
        'over_threshold',  # Target variable
        'player_threshold',  # Keep for reference but not as feature
        'gameId', 'personId',  # ID features
        'firstName', 'lastName', 'fullName',  # Name features (keep fullName for grouping)
        'gameDate'  # Keep for temporal sorting but not as feature
    ]

    # Additional features that might contain game result information
    potentially_leaky_features = [
        'fieldGoalsMade', 'fieldGoalsAttempted', 'fieldGoalsPercentage',  # Current game shooting
        'threePointersMade', 'threePointersAttempted', 'threePointersPercentage',  # Current game 3pt
        'freeThrowsMade', 'freeThrowsAttempted', 'freeThrowsPercentage',  # Current game FT
        'assists', 'reboundsTotal', 'steals', 'blocks', 'turnovers',  # Current game stats
        'numMinutes',  # Current game minutes (could indicate game flow)
        'home', 'win'  # Game outcome information
    ]

    print(f"Removing {len(leakage_features)} definite leakage features")
    print(f"Removing {len(potentially_leaky_features)} potentially leaky features")

    # Keep only features that are available before the game starts
    safe_features = [
        # Rolling averages (from past games)
        'rolling_3g_points', 'rolling_5g_points', 'rolling_10g_points',
        'rolling_3g_numMinutes', 'rolling_5g_numMinutes', 'rolling_10g_numMinutes',
        'rolling_3g_fieldGoalsMade', 'rolling_5g_fieldGoalsMade', 'rolling_10g_fieldGoalsMade',
        'rolling_3g_fieldGoalsAttempted', 'rolling_5g_fieldGoalsAttempted', 'rolling_10g_fieldGoalsAttempted',
        'rolling_3g_assists', 'rolling_5g_assists', 'rolling_10g_assists',
        'rolling_3g_fg_pct', 'rolling_5g_fg_pct', 'rolling_10g_fg_pct',
        'rolling_3g_3pt_pct', 'rolling_5g_3pt_pct', 'rolling_10g_3pt_pct',

        # Rest and schedule features (available before game)
        'days_rest', 'is_back_to_back', 'games_last_7d', 'games_last_14d',
        'avg_minutes_last_3g', 'avg_minutes_last_7g',

        # Momentum and trend features (from past games)
        'over_streak_last_3g', 'over_streak_last_5g',
        'vs_threshold_trend_3g', 'vs_threshold_trend_5g',
        'minutes_trend_3g', 'minutes_trend_5g',
        'hot_streak', 'cold_streak',

        # Contextual features (known before game)
        'days_since_season_start', 'season_progression', 'month',
        'day_of_week', 'is_weekend', 'home_game',

        # Advanced features (from historical data)
        'points_volatility_5g', 'points_volatility_10g',
        'recent_fg_hot', 'recent_3pt_hot', 'good_form',

        # Team and opponent information (known before game)
        'playerteamCity', 'playerteamName', 'opponentteamCity', 'opponentteamName',
        'gameType', 'gameLabel'
    ]

    # Keep categorical features for encoding
    categorical_features = ['rest_category', 'season_month', 'age_category', 'career_stage']
    safe_features.extend(categorical_features)

    # Filter to features that actually exist in the data
    available_safe_features = [f for f in safe_features if f in data.columns]

    # Always keep target variable and key identifiers
    model_features = available_safe_features + ['fullName', 'gameDate', 'over_threshold', 'player_threshold']

    clean_data = data[model_features].copy()

    print(f"✓ Kept {len(available_safe_features)} safe features for modeling")
    print(f"✓ Removed {data.shape[1] - len(model_features)} potentially leaky features")

    return clean_data, available_safe_features

def prepare_ml_features(clean_data, feature_cols):
    """Prepare features for ML training."""

    print("\nPreparing features for ML training...")

    # Create temporal split (80/20)
    clean_data_sorted = clean_data.sort_values('gameDate')
    split_idx = int(len(clean_data_sorted) * 0.8)

    train_data = clean_data_sorted.iloc[:split_idx].copy()
    test_data = clean_data_sorted.iloc[split_idx:].copy()

    print(f"✓ Train: {len(train_data):,} games")
    print(f"✓ Test: {len(test_data):,} games")

    # Prepare feature matrices
    X_train = train_data[feature_cols].copy()
    y_train = train_data['over_threshold'].copy()
    X_test = test_data[feature_cols].copy()
    y_test = test_data['over_threshold'].copy()

    # Handle categorical features
    categorical_features = []
    for col in feature_cols:
        if X_train[col].dtype == 'object' or str(X_train[col].dtype) == 'category':
            categorical_features.append(col)

    print(f"✓ Found {len(categorical_features)} categorical features")

    # Encode categorical features
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()

        # Handle missing values
        train_vals = X_train[col].fillna('missing').astype(str)
        test_vals = X_test[col].fillna('missing').astype(str)

        # Fit on combined data
        combined_vals = pd.concat([train_vals, test_vals])
        le.fit(combined_vals)

        X_train[col] = le.transform(train_vals)
        X_test[col] = le.transform(test_vals)

        label_encoders[col] = le

    # Handle missing values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Ensure all numeric
    for col in feature_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)

    print(f"✓ Final feature matrix: {X_train.shape}")
    print(f"✓ Missing values: Train={X_train.isnull().sum().sum()}, Test={X_test.isnull().sum().sum()}")

    return X_train, X_test, y_train, y_test, label_encoders

def train_clean_models(X_train, y_train, X_test, y_test):
    """Train models without data leakage."""

    print("\n=== TRAINING CLEAN MODELS ===")

    results = []

    # 1. Logistic Regression
    print("\n--- LOGISTIC REGRESSION (CLEAN) ---")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr.fit(X_train_scaled, y_train)

    lr_pred = lr.predict(X_test_scaled)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]

    lr_metrics = {
        'model': 'Logistic Regression (Clean)',
        'accuracy': accuracy_score(y_test, lr_pred),
        'precision': precision_score(y_test, lr_pred),
        'recall': recall_score(y_test, lr_pred),
        'f1_score': f1_score(y_test, lr_pred),
        'roc_auc': roc_auc_score(y_test, lr_proba),
        'log_loss': log_loss(y_test, lr_proba)
    }

    print(f"  Accuracy: {lr_metrics['accuracy']:.3f}")
    print(f"  F1 Score: {lr_metrics['f1_score']:.3f}")
    print(f"  ROC AUC: {lr_metrics['roc_auc']:.3f}")

    results.append(lr_metrics)

    # 2. Random Forest
    print("\n--- RANDOM FOREST (CLEAN) ---")

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5,
        random_state=42, class_weight='balanced', n_jobs=-1
    )
    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    rf_metrics = {
        'model': 'Random Forest (Clean)',
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred),
        'recall': recall_score(y_test, rf_pred),
        'f1_score': f1_score(y_test, rf_pred),
        'roc_auc': roc_auc_score(y_test, rf_proba),
        'log_loss': log_loss(y_test, rf_proba)
    }

    print(f"  Accuracy: {rf_metrics['accuracy']:.3f}")
    print(f"  F1 Score: {rf_metrics['f1_score']:.3f}")
    print(f"  ROC AUC: {rf_metrics['roc_auc']:.3f}")

    results.append(rf_metrics)

    return results, lr, rf, scaler

def main():
    """Main workflow for clean ML models."""

    # Load data
    data = load_and_prepare_data()

    # Remove leakage features
    clean_data, safe_features = remove_data_leakage_features(data)

    # Prepare for ML
    X_train, X_test, y_train, y_test, encoders = prepare_ml_features(clean_data, safe_features)

    # Train clean models
    results, lr_model, rf_model, scaler = train_clean_models(X_train, y_train, X_test, y_test)

    # Compare results
    print("\n=== CLEAN MODEL COMPARISON ===")
    results_df = pd.DataFrame(results)
    print(results_df[['model', 'accuracy', 'f1_score', 'roc_auc']].to_string(index=False))

    # Save results
    os.makedirs('../data/processed', exist_ok=True)
    results_df.to_csv('../data/processed/clean_ml_results.csv', index=False)

    # Compare to baseline
    baseline_accuracy = 0.639  # Our best baseline (rolling 5-game)
    best_ml_accuracy = results_df['accuracy'].max()

    print("\n=== FINAL RESULTS (DATA LEAKAGE FIXED) ===")
    print(f"✅ Best clean ML accuracy: {best_ml_accuracy:.3f}")
    print(f"✅ Rolling 5-game baseline: 0.639")
    print(f"✅ Improvement over baseline: {best_ml_accuracy - baseline_accuracy:+.3f}")

    if best_ml_accuracy > baseline_accuracy:
        print("🎯 SUCCESS: Clean ML model beats baseline!")
    else:
        print("📊 INFO: Baseline remains competitive - need more features")

    print(f"✅ Models trained without data leakage")
    print(f"✅ Results saved and ready for deployment")

    return results_df

if __name__ == "__main__":
    main()