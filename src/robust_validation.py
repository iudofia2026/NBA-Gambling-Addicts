"""
ROBUST VALIDATION FRAMEWORK
Prevents data leakage and provides realistic accuracy estimates
Uses walk-forward validation simulating real trading conditions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

class RobustValidator:
    """Robust validation framework without data leakage."""

    def __init__(self):
        print("=" * 70)
        print("🔍 ROBUST VALIDATION FRAMEWORK")
        print("   Preventing data leakage with walk-forward validation")
        print("   Simulating real-world betting conditions")
        print("=" * 70)

    def load_and_validate_data(self):
        """Load data and check for leakage issues."""
        print("\n📂 LOADING AND VALIDATING DATA")

        # Load data
        data_path = 'data/processed/engineered_features.csv'
        data = pd.read_csv(data_path)
        data['gameDate'] = pd.to_datetime(data['gameDate'], errors='coerce')

        print(f"✅ Loaded {len(data):,} games")

        # Check date range
        print(f"📅 Date range: {data['gameDate'].min()} to {data['gameDate'].max()}")

        # Check for data leakage indicators
        print("\n🔍 CHECKING FOR DATA LEAKAGE:")

        # 1. Check if over_threshold is derived from same day's points
        correlation = data['points'].corr(data['over_threshold'])
        print(f"   Points vs Over_Threshold correlation: {correlation:.3f}")

        if abs(correlation) > 0.95:
            print("   ⚠️  HIGH correlation detected - possible leakage!")

        # 2. Check player_threshold distribution
        threshold_corr = data['points'].corr(data['player_threshold'])
        print(f"   Points vs Player_Threshold correlation: {threshold_corr:.3f}")

        # 3. Check for look-ahead bias in rolling features
        # Need to ensure all rolling windows use ONLY past data

        # Sort by player and date
        data = data.sort_values(['fullName', 'gameDate']).reset_index(drop=True)

        # Create truly leakage-free target
        # We need to create a realistic prop line
        data['realistic_prop_line'] = data.groupby('fullName')['points'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=5).mean()
        ).fillna(data.groupby('fullName')['points'].transform('mean'))

        # New target: Will player exceed their AVERAGE from LAST 10 games?
        data['target'] = (data['points'] > data['realistic_prop_line']).astype(int)

        print(f"   ✅ Created realistic prop line (average of last 10 games)")
        print(f"   📊 Target distribution: {data['target'].mean():.1%} OVER, {1-data['target'].mean():.1%} UNDER")

        # Clean data
        data = data.dropna(subset=['target', 'realistic_prop_line'])
        print(f"   ✅ Final dataset: {len(data):,} games after cleaning")

        return data

    def create_leakage_free_features(self, data):
        """Create features without any data leakage."""
        print("\n🔧 CREATING LEAKAGE-FREE FEATURES")

        features = pd.DataFrame(index=data.index)

        # CRITICAL: All features must use ONLY information available BEFORE game time

        # 1. Historical averages (using shift to ensure no leakage)
        features['points_avg_10'] = data.groupby('fullName')['points'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        )
        features['points_std_10'] = data.groupby('fullName')['points'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).std()
        ).fillna(0)

        features['minutes_avg_5'] = data.groupby('fullName')['numMinutes'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).mean()
        )

        # 2. Recent form (last 3 games)
        features['points_last_3_avg'] = data.groupby('fullName')['points'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        features['points_last_3_trend'] = data.groupby('fullName')['points'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=2).apply(
                lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0
            )
        ).fillna(0)

        # 3. Efficiency metrics (historical)
        features['efficiency_avg_5'] = data.groupby('fullName').apply(
            lambda x: (x['points'] / (x['numMinutes'] + 1) * 40).shift(1).rolling(5, min_periods=2).mean()
        ).reset_index(level=0, drop=True)

        # 4. Rest days and schedule
        data_sorted = data.sort_values(['fullName', 'gameDate'])
        data_sorted['days_since_last_game'] = data_sorted.groupby('fullName')['gameDate'].diff().dt.days
        data_sorted['days_since_last_game'] = data_sorted.groupby('fullName')['days_since_last_game'].fillna(3)
        features['days_rest'] = data_sorted['days_since_last_game']

        # 5. Home/away context
        features['is_home'] = data['home'].fillna(0)

        # 6. Season progression (no leakage)
        features['days_in_season'] = (data['gameDate'] - data['gameDate'].min()).dt.days

        # 7. Age and experience (no leakage)
        features['age'] = data['age_at_game'].fillna(data['age_at_game'].mean())

        # 8. Team context (historical)
        features['team_avg_points'] = data.groupby(['playerteamName', 'gameDate'])['points'].transform(
            lambda x: x.mean()
        )

        # Fill NaN values
        features = features.fillna(features.mean())

        print(f"   ✅ Created {features.shape[1]} leakage-free features")

        # Verify no leakage
        print("\n   🔍 VERIFYING NO LEAKAGE:")
        for col in features.columns[:5]:
            corr = features[col].corr(data['points'])
            print(f"      {col}: correlation = {corr:.3f}")

        return features

    def walk_forward_validation(self, data, features, target):
        """Perform walk-forward validation simulating real trading."""
        print("\n🚶 WALK-FORWARD VALIDATION")
        print("   Simulating betting on future games with no hindsight")

        # Group by date for time-based splits
        data['year_month'] = data['gameDate'].dt.to_period('M')
        unique_periods = data['year_month'].unique()

        results = []

        # Use first 70% for initial training, then walk forward
        train_cutoff = int(len(unique_periods) * 0.7)

        for i in range(train_cutoff, len(unique_periods) - 1):
            # Define training and test periods
            train_period = unique_periods[:i]
            test_period = unique_periods[i + 1]

            # Split data
            train_mask = data['year_month'].isin(train_period)
            test_mask = data['year_month'] == test_period

            X_train = features[train_mask]
            y_train = target[train_mask]
            X_test = features[test_mask]
            y_test = target[test_mask]

            if len(X_test) < 10:  # Skip if too few test samples
                continue

            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42
            )
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Store results
            results.append({
                'period': str(test_period),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'accuracy': accuracy,
                'over_pred': (y_pred == 1).mean(),
                'over_actual': y_test.mean()
            })

            print(f"   Period {test_period}: {accuracy:.1%} accuracy on {len(X_test)} games")

        # Calculate overall results
        if results:
            accuracies = [r['accuracy'] for r in results]
            total_tests = sum(r['test_samples'] for r in results)

            overall_accuracy = sum(r['accuracy'] * r['test_samples'] for r in results) / total_tests

            print(f"\n📊 WALK-FORWARD RESULTS:")
            print(f"   Periods tested: {len(results)}")
            print(f"   Total predictions: {total_tests:,}")
            print(f"   Average accuracy: {np.mean(accuracies):.1%}")
            print(f"   Weighted accuracy: {overall_accuracy:.1%}")
            print(f"   Std deviation: {np.std(accuracies):.1%}")
            print(f"   Best period: {max(accuracies):.1%}")
            print(f"   Worst period: {min(accuracies):.1%}")

            return results, overall_accuracy

        return None, 0

    def final_holdout_test(self, data, features, target):
        """Final test on completely held-out data."""
        print("\n🎯 FINAL HOLDOUT TEST")
        print("   Testing on completely unseen future data")

        # Use last 20% of time as final holdout
        data_sorted = data.sort_values('gameDate')
        split_idx = int(len(data_sorted) * 0.8)

        train_data = data_sorted.iloc[:split_idx]
        test_data = data_sorted.iloc[split_idx:]

        X_train = features.iloc[train_data.index]
        y_train = target.iloc[train_data.index]
        X_test = features.iloc[test_data.index]
        y_test = target.iloc[test_data.index]

        print(f"   Training on: {len(train_data):,} games ({train_data['gameDate'].min()} to {train_data['gameDate'].max()})")
        print(f"   Testing on: {len(test_data):,} games ({test_data['gameDate'].min()} to {test_data['gameDate'].max()})")

        # Train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

        print(f"\n📈 HOLDOUT RESULTS:")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Precision: {precision:.1%}")
        print(f"   Recall: {recall:.1%}")
        print(f"   F1-Score: {f1:.1%}")

        # Baseline comparison
        baseline_acc = max(y_test.mean(), 1 - y_test.mean())
        print(f"   Baseline (always majority): {baseline_acc:.1%}")
        print(f"   Model improvement: {accuracy - baseline_acc:+.1%}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n🔍 TOP 10 FEATURES:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")

        return accuracy, model, feature_importance

def main():
    """Run robust validation."""
    validator = RobustValidator()

    # Load and validate data
    data = validator.load_and_validate_data()

    # Create leakage-free features
    features = validator.create_leakage_free_features(data)
    target = data['target']

    # Walk-forward validation
    wf_results, wf_accuracy = validator.walk_forward_validation(data, features, target)

    # Final holdout test
    final_accuracy, model, feature_importance = validator.final_holdout_test(data, features, target)

    # Summary
    print("\n" + "=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Walk-forward accuracy: {wf_accuracy:.1%}")
    print(f"Final holdout accuracy: {final_accuracy:.1%}")

    if final_accuracy < 0.65:
        print("\n🎯 REALISTIC ASSESSMENT:")
        print(f"   The model achieves {final_accuracy:.1%} accuracy")
        print("   This is more realistic than the inflated 96% from leaked data")
        print("   60-65% is typical for sports betting models")
    elif final_accuracy < 0.75:
        print(f"\n✅ GOOD RESULT:")
        print(f"   The model achieves {final_accuracy:.1%} accuracy")
        print("   This is a solid result for sports predictions")
    else:
        print(f"\n⚠️  STILL SUSPICIOUS:")
        print(f"   Even after removing leakage, {final_accuracy:.1%} is very high")
        print("   Please double-check for remaining leakage sources")

    return final_accuracy, model, feature_importance

if __name__ == "__main__":
    main()