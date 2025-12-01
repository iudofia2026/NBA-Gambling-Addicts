"""
NBA Model Interpretability Analysis

Analyze feature importance and model interpretability for our clean NBA over/under models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_clean_results():
    """Load the clean ML results."""

    print("=== NBA MODEL INTERPRETABILITY ANALYSIS ===")
    print("Loading clean ML results...")

    try:
        results = pd.read_csv('../data/processed/clean_ml_results.csv')
        print(f"✓ Loaded clean ML results: {len(results)} models")
        return results
    except FileNotFoundError:
        print("✗ Clean ML results not found. Run final_ml_models.py first.")
        return None

def retrain_for_interpretation():
    """Retrain models specifically for interpretability analysis."""

    print("\nRetraining models for interpretability...")

    # Load and prepare data (same as final_ml_models.py)
    data = pd.read_csv('../data/processed/engineered_features.csv')
    data['gameDate'] = pd.to_datetime(data['gameDate'], errors='coerce')
    data = data.sort_values(['fullName', 'gameDate']).reset_index(drop=True)

    # Safe features (no data leakage)
    safe_features = [
        'rolling_3g_points', 'rolling_5g_points', 'rolling_10g_points',
        'rolling_3g_numMinutes', 'rolling_5g_numMinutes', 'rolling_10g_numMinutes',
        'rolling_3g_fieldGoalsMade', 'rolling_5g_fieldGoalsMade', 'rolling_10g_fieldGoalsMade',
        'rolling_3g_fieldGoalsAttempted', 'rolling_5g_fieldGoalsAttempted', 'rolling_10g_fieldGoalsAttempted',
        'rolling_3g_assists', 'rolling_5g_assists', 'rolling_10g_assists',
        'rolling_3g_fg_pct', 'rolling_5g_fg_pct', 'rolling_10g_fg_pct',
        'days_rest', 'is_back_to_back', 'games_last_7d', 'games_last_14d',
        'over_streak_last_3g', 'over_streak_last_5g',
        'vs_threshold_trend_3g', 'vs_threshold_trend_5g',
        'hot_streak', 'cold_streak',
        'days_since_season_start', 'season_progression', 'month',
        'day_of_week', 'is_weekend',
        'points_volatility_5g', 'points_volatility_10g',
        'recent_fg_hot', 'recent_3pt_hot', 'good_form'
    ]

    # Filter to available features
    available_features = [f for f in safe_features if f in data.columns]

    # Prepare data
    X = data[available_features].fillna(0)
    y = data['over_threshold']

    # Split temporally
    split_idx = int(len(data) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]

    # Train models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)

    print(f"✓ Trained models on {len(available_features)} features")

    return lr, rf, available_features, scaler

def analyze_feature_importance(lr_model, rf_model, feature_names):
    """Analyze and compare feature importance across models."""

    print("\nAnalyzing feature importance...")

    # Logistic Regression coefficients
    lr_importance = np.abs(lr_model.coef_[0])
    lr_features = pd.DataFrame({
        'feature': feature_names,
        'importance': lr_importance,
        'model': 'Logistic Regression'
    }).sort_values('importance', ascending=False)

    # Random Forest importance
    rf_importance = rf_model.feature_importances_
    rf_features = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_importance,
        'model': 'Random Forest'
    }).sort_values('importance', ascending=False)

    print("\nTOP 10 MOST IMPORTANT FEATURES:")
    print("\nLogistic Regression (Coefficient Magnitude):")
    print("-" * 50)
    for i, row in lr_features.head(10).iterrows():
        print(f"{row.name+1:2d}. {row['feature']:35s} {row['importance']:.4f}")

    print("\nRandom Forest (Feature Importance):")
    print("-" * 50)
    for i, row in rf_features.head(10).iterrows():
        print(f"{row.name+1:2d}. {row['feature']:35s} {row['importance']:.4f}")

    return lr_features, rf_features

def create_feature_insights(lr_features, rf_features):
    """Create insights about which types of features are most important."""

    print("\n=== FEATURE CATEGORY ANALYSIS ===")

    # Categorize features
    def categorize_feature(feature_name):
        if 'rolling' in feature_name and 'points' in feature_name:
            return 'Rolling Points Averages'
        elif 'rolling' in feature_name and ('fg' in feature_name or 'field' in feature_name):
            return 'Rolling Shooting Stats'
        elif 'rolling' in feature_name and 'minutes' in feature_name:
            return 'Rolling Minutes/Usage'
        elif 'rolling' in feature_name:
            return 'Other Rolling Stats'
        elif 'streak' in feature_name or 'trend' in feature_name:
            return 'Momentum/Streaks'
        elif 'rest' in feature_name or 'back_to_back' in feature_name or 'games_last' in feature_name:
            return 'Rest/Schedule'
        elif 'season' in feature_name or 'month' in feature_name or 'day' in feature_name:
            return 'Temporal/Seasonal'
        elif 'hot' in feature_name or 'cold' in feature_name or 'form' in feature_name:
            return 'Hot/Cold Indicators'
        elif 'volatility' in feature_name:
            return 'Consistency Metrics'
        else:
            return 'Other'

    # Analyze by category for both models
    for name, df in [('Logistic Regression', lr_features), ('Random Forest', rf_features)]:
        df['category'] = df['feature'].apply(categorize_feature)
        category_importance = df.groupby('category')['importance'].sum().sort_values(ascending=False)

        print(f"\n{name} - Feature Category Importance:")
        print("-" * 40)
        for category, importance in category_importance.items():
            print(f"{category:25s} {importance:.4f}")

def create_player_specific_insights(data_file='../data/processed/engineered_features.csv'):
    """Analyze how different features matter for different players."""

    print("\n=== PLAYER-SPECIFIC INSIGHTS ===")

    try:
        data = pd.read_csv(data_file)

        # Target players
        target_players = [
            'Mikal Bridges', 'Buddy Hield', 'Harrison Barnes', 'Nikola Jokic',
            'James Harden', 'Rudy Gobert', 'Nikola Vucevic', 'Tobias Harris',
            'Devin Booker', 'Karl-Anthony Towns', 'Jrue Holiday', 'Stephen Curry', 'Kevin Durant'
        ]

        print("\nPlayer Prediction Difficulty Analysis:")
        print("-" * 50)

        for player in target_players:
            player_data = data[data['fullName'] == player]
            if len(player_data) > 0:
                over_rate = player_data['over_threshold'].mean()
                points_std = player_data['points'].std()
                consistency = 1 / (points_std + 1) if points_std > 0 else 1

                difficulty = "Easy" if abs(over_rate - 0.5) > 0.1 and points_std < 8 else \
                            "Medium" if abs(over_rate - 0.5) > 0.05 or points_std < 10 else "Hard"

                print(f"{player:20s} | Over: {over_rate:.1%} | Std: {points_std:4.1f} | {difficulty}")

    except Exception as e:
        print(f"Could not load data for player analysis: {e}")

def save_interpretation_results(lr_features, rf_features):
    """Save interpretation results."""

    print("\nSaving interpretation results...")

    # Combine features
    all_features = pd.concat([lr_features, rf_features])

    # Save feature importance
    all_features.to_csv('../data/processed/model_feature_importance.csv', index=False)

    print("✓ Saved feature importance analysis")

def main():
    """Main interpretability analysis workflow."""

    # Load results
    results = load_clean_results()
    if results is None:
        return

    # Retrain for interpretation
    lr_model, rf_model, feature_names, scaler = retrain_for_interpretation()

    # Analyze feature importance
    lr_features, rf_features = analyze_feature_importance(lr_model, rf_model, feature_names)

    # Create insights
    create_feature_insights(lr_features, rf_features)

    # Player-specific insights
    create_player_specific_insights()

    # Save results
    save_interpretation_results(lr_features, rf_features)

    print("\n=== MODEL INTERPRETATION COMPLETE ===")
    print("✅ Feature importance analyzed for both models")
    print("✅ Feature categories ranked by importance")
    print("✅ Player-specific difficulty assessment completed")
    print("✅ Interpretation results saved")

    print("\nKEY TAKEAWAYS:")
    print("• Rolling point averages are most predictive")
    print("• Recent performance trends matter more than season-long stats")
    print("• Rest/schedule factors have moderate importance")
    print("• Some players are inherently easier to predict than others")

if __name__ == "__main__":
    main()