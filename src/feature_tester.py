"""
Feature Testing Framework for NBA Prediction System
Tests experimental features before adding to final system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FeatureTester:
    """Framework to test new features before integration."""

    def __init__(self, historical_data_path='../data/processed/engineered_features.csv'):
        """Initialize with historical data."""
        self.data = pd.read_csv(historical_data_path)
        self.data['gameDate'] = pd.to_datetime(self.data['gameDate'], errors='coerce')
        self.data = self.data.sort_values('gameDate')

        # Create train/test split (temporal)
        cutoff_date = self.data['gameDate'].quantile(0.8)
        self.train_data = self.data[self.data['gameDate'] < cutoff_date]
        self.test_data = self.data[self.data['gameDate'] >= cutoff_date]

        print(f"FeatureTester initialized:")
        print(f"  - Training data: {len(self.train_data):,} games")
        print(f"  - Test data: {len(self.test_data):,} games")
        print(f"  - Test cutoff: {cutoff_date.date()}")

    def calculate_baseline_accuracy(self):
        """Calculate baseline accuracy using simple rolling averages."""
        print("\n=== Calculating Baseline Accuracy ===")

        correct = 0
        total = 0

        for _, row in self.test_data.iterrows():
            # Simple baseline: predict OVER if rolling 5-game average > threshold
            if pd.notna(row['rolling_5g_points']) and pd.notna(row['player_threshold']):
                prediction = 1 if row['rolling_5g_points'] > row['player_threshold'] else 0
                actual = row.get('over_threshold', 0)

                if pd.notna(actual):
                    if prediction == actual:
                        correct += 1
                    total += 1

        baseline_accuracy = correct / total if total > 0 else 0
        print(f"Baseline Accuracy (5-game rolling avg): {baseline_accuracy:.1%}")
        print(f"Based on {total:,} test predictions")

        return baseline_accuracy

    def test_feature_improvement(self, feature_function, feature_name, sample_players=20):
        """Test if a new feature improves prediction accuracy."""
        print(f"\n=== Testing Feature: {feature_name} ===")

        # Select sample players with sufficient data
        player_counts = self.train_data['fullName'].value_counts()
        test_players = player_counts[player_counts >= 30].head(sample_players).index.tolist()

        improvements = []

        for player in test_players:
            player_test_data = self.test_data[self.test_data['fullName'] == player]
            player_train_data = self.train_data[self.train_data['fullName'] == player]

            if len(player_test_data) < 5:
                continue

            # Test with and without feature
            baseline_correct = 0
            feature_correct = 0
            total = 0

            for _, row in player_test_data.iterrows():
                if pd.notna(row['rolling_5g_points']) and pd.notna(row['player_threshold']):
                    # Baseline prediction
                    baseline_pred = 1 if row['rolling_5g_points'] > row['player_threshold'] else 0

                    # Feature-enhanced prediction
                    feature_adjustment = feature_function(player_train_data, row)
                    feature_value = row['rolling_5g_points'] + feature_adjustment
                    feature_pred = 1 if feature_value > row['player_threshold'] else 0

                    actual = row.get('over_threshold', 0)

                    if pd.notna(actual):
                        if baseline_pred == actual:
                            baseline_correct += 1
                        if feature_pred == actual:
                            feature_correct += 1
                        total += 1

            if total > 0:
                baseline_acc = baseline_correct / total
                feature_acc = feature_correct / total
                improvement = feature_acc - baseline_acc
                improvements.append(improvement)

        if improvements:
            avg_improvement = np.mean(improvements)
            positive_improvements = [i for i in improvements if i > 0]
            print(f"Average accuracy improvement: {avg_improvement:+.2%}")
            print(f"Features helped in {len(positive_improvements)}/{len(improvements)} cases")

            return avg_improvement, improvements
        else:
            print("Insufficient data for testing")
            return 0, []

    def generate_test_report(self, test_results):
        """Generate a comprehensive test report."""
        print("\n" + "="*60)
        print("FEATURE TESTING SUMMARY REPORT")
        print("="*60)

        baseline = test_results.get('baseline', 0)
        print(f"\nBaseline Accuracy: {baseline:.1%}")
        print(f"\nFeature Improvements:")

        for feature, result in test_results.items():
            if feature != 'baseline':
                improvement, improvements = result
                if improvement:
                    print(f"  {feature:30}: {improvement:+.2%} ({len([i for i in improvements if i > 0])} positive)")
                else:
                    print(f"  {feature:30}: No data")

        # Recommend features that improve accuracy
        recommended = [
            feature for feature, (improvement, _) in test_results.items()
            if feature != 'baseline' and improvement and improvement > 0.01
        ]

        print(f"\n✅ RECOMMENDED FEATURES (improve accuracy > 1%):")
        for feature in recommended:
            improvement, _ = test_results[feature]
            print(f"  - {feature}: +{improvement:.2%}")