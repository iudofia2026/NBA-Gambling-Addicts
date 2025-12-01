"""
Test Feature Migration Script
Tests each experimental feature for performance improvement
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def test_seasonal_trends_feature():
    """Test seasonal trends feature improvement."""
    print("\n" + "="*60)
    print("TESTING SEASONAL TRENDS FEATURE (Iteration 6)")
    print("="*60)

    # Load data
    data = pd.read_csv('../data/processed/engineered_features.csv')
    data['gameDate'] = pd.to_datetime(data['gameDate'], errors='coerce')
    data = data.sort_values('gameDate')

    # Create train/test split
    cutoff_date = data['gameDate'].quantile(0.8)
    train_data = data[data['gameDate'] < cutoff_date]
    test_data = data[data['gameDate'] >= cutoff_date]

    improvements = []

    for player in test_data['fullName'].unique()[:20]:  # Test 20 players
        player_train = train_data[train_data['fullName'] == player]
        player_test = test_data[test_data['fullName'] == player]

        if len(player_train) < 30 or len(player_test) < 5:
            continue

        for _, row in player_test.iterrows():
            if pd.notna(row['rolling_5g_points']) and pd.notna(row['player_threshold']):
                # Baseline prediction
                baseline_pred = row['rolling_5g_points']

                # Seasonal adjustment
                current_month = pd.to_datetime(row['gameDate']).month
                player_train['month'] = pd.to_datetime(player_train['gameDate']).dt.month
                monthly_avg = player_train.groupby('month')['points'].mean()

                if current_month in monthly_avg.index:
                    seasonal_factor = monthly_avg[current_month] / monthly_avg.mean()
                    seasonal_pred = baseline_pred * seasonal_factor
                else:
                    seasonal_pred = baseline_pred

                # Compare accuracy
                actual = row['over_threshold']
                baseline_correct = (1 if baseline_pred > row['player_threshold'] else 0) == actual
                seasonal_correct = (1 if seasonal_pred > row['player_threshold'] else 0) == actual

                if baseline_correct != seasonal_correct:
                    if seasonal_correct:
                        improvements.append(1)
                    else:
                        improvements.append(-1)

    if improvements:
        avg_improvement = np.mean(improvements) * 0.01  # Convert to percentage
        positive_rate = sum(1 for i in improvements if i > 0) / len(improvements)

        print(f"✅ Seasonal Trends Feature Results:")
        print(f"   - Average accuracy improvement: {avg_improvement:+.2%}")
        print(f"   - Positive impact rate: {positive_rate:.1%}")
        print(f"   - Recommendation: {'ADD' if avg_improvement > 0.01 else 'SKIP'}")

        return avg_improvement
    else:
        print("❌ Insufficient data for seasonal trends test")
        return 0

def test_peak_performance_feature():
    """Test peak performance analysis feature."""
    print("\n" + "="*60)
    print("TESTING PEAK PERFORMANCE FEATURE (Iteration 7)")
    print("="*60)

    data = pd.read_csv('../data/processed/engineered_features.csv')
    data['gameDate'] = pd.to_datetime(data['gameDate'], errors='coerce')
    data = data.sort_values('gameDate')

    cutoff_date = data['gameDate'].quantile(0.8)
    train_data = data[data['gameDate'] < cutoff_date]
    test_data = data[data['gameDate'] >= cutoff_date]

    improvements = []

    for player in test_data['fullName'].unique()[:20]:
        player_train = train_data[train_data['fullName'] == player]
        player_test = test_data[test_data['fullName'] == player]

        if len(player_train) < 30 or len(player_test) < 5:
            continue

        # Calculate peak performance metrics
        points_mean = player_train['points'].mean()
        top_10_percentile = player_train['points'].quantile(0.90)
        recent_form = player_train.tail(5)['points'].mean()

        for _, row in player_test.iterrows():
            if pd.notna(row['rolling_5g_points']) and pd.notna(row['player_threshold']):
                # Baseline
                baseline_pred = row['rolling_5g_points']

                # Peak performance adjustment
                peak_factor = (top_10_percentile - points_mean) / points_mean if points_mean > 0 else 0
                form_factor = (recent_form - points_mean) / points_mean if points_mean > 0 else 0
                peak_adjusted = baseline_pred * (1 + (peak_factor + form_factor) * 0.3)

                # Compare
                actual = row['over_threshold']
                baseline_correct = (1 if baseline_pred > row['player_threshold'] else 0) == actual
                peak_correct = (1 if peak_adjusted > row['player_threshold'] else 0) == actual

                if baseline_correct != peak_correct:
                    if peak_correct:
                        improvements.append(1)
                    else:
                        improvements.append(-1)

    if improvements:
        avg_improvement = np.mean(improvements) * 0.01
        positive_rate = sum(1 for i in improvements if i > 0) / len(improvements)

        print(f"✅ Peak Performance Feature Results:")
        print(f"   - Average accuracy improvement: {avg_improvement:+.2%}")
        print(f"   - Positive impact rate: {positive_rate:.1%}")
        print(f"   - Recommendation: {'ADD' if avg_improvement > 0.01 else 'SKIP'}")

        return avg_improvement
    else:
        print("❌ Insufficient data for peak performance test")
        return 0

def test_team_dynamics_feature():
    """Test enhanced team dynamics feature."""
    print("\n" + "="*60)
    print("TESTING TEAM DYNAMICS FEATURE (Iteration 8)")
    print("="*60)

    data = pd.read_csv('../data/processed/engineered_features.csv')
    data['gameDate'] = pd.to_datetime(data['gameDate'], errors='coerce')
    data = data.sort_values('gameDate')

    cutoff_date = data['gameDate'].quantile(0.8)
    train_data = data[data['gameDate'] < cutoff_date]
    test_data = data[data['gameDate'] >= cutoff_date]

    improvements = []

    for player in test_data['fullName'].unique()[:20]:
        player_train = train_data[train_data['fullName'] == player]
        player_test = test_data[test_data['fullName'] == player]

        if len(player_train) < 30 or len(player_test) < 5:
            continue

        for _, row in player_test.iterrows():
            if pd.notna(row['rolling_5g_points']) and pd.notna(row['player_threshold']):
                # Baseline
                baseline_pred = row['rolling_5g_points']

                # Team dynamics adjustment
                team_name = row.get('playerteamName', row.get('opponentteamName', ''))
                if team_name:
                    # Calculate team momentum
                    team_train = train_data[
                        (train_data['playerteamName'] == team_name) |
                        (train_data['opponentteamName'] == team_name)
                    ]

                    if len(team_train) >= 10:
                        recent_team = team_train.tail(5)['points'].mean()
                        older_team = team_train.head(len(team_train) - 5)['points'].mean()
                        team_momentum = (recent_team - older_team) / older_team if older_team > 0 else 0

                        # Player's role
                        player_team_train = team_train[team_train['fullName'] == player]
                        if len(player_team_train) > 0:
                            usage_share = player_team_train['points'].mean() / recent_team if recent_team > 0 else 0.2
                            team_adjusted = baseline_pred * (1 + team_momentum * usage_share * 0.3)
                        else:
                            team_adjusted = baseline_pred
                    else:
                        team_adjusted = baseline_pred
                else:
                    team_adjusted = baseline_pred

                # Compare
                actual = row['over_threshold']
                baseline_correct = (1 if baseline_pred > row['player_threshold'] else 0) == actual
                team_correct = (1 if team_adjusted > row['player_threshold'] else 0) == actual

                if baseline_correct != team_correct:
                    if team_correct:
                        improvements.append(1)
                    else:
                        improvements.append(-1)

    if improvements:
        avg_improvement = np.mean(improvements) * 0.01
        positive_rate = sum(1 for i in improvements if i > 0) / len(improvements)

        print(f"✅ Team Dynamics Feature Results:")
        print(f"   - Average accuracy improvement: {avg_improvement:+.2%}")
        print(f"   - Positive impact rate: {positive_rate:.1%}")
        print(f"   - Recommendation: {'ADD' if avg_improvement > 0.01 else 'SKIP'}")

        return avg_improvement
    else:
        print("❌ Insufficient data for team dynamics test")
        return 0

def test_situational_pressure_feature():
    """Test situational pressure/clutch feature."""
    print("\n" + "="*60)
    print("TESTING SITUATIONAL PRESSURE FEATURE (Iteration 9)")
    print("="*60)

    data = pd.read_csv('../data/processed/engineered_features.csv')
    data['gameDate'] = pd.to_datetime(data['gameDate'], errors='coerce')
    data = data.sort_values('gameDate')

    cutoff_date = data['gameDate'].quantile(0.8)
    train_data = data[data['gameDate'] < cutoff_date]
    test_data = data[data['gameDate'] >= cutoff_date]

    improvements = []

    for player in test_data['fullName'].unique()[:20]:
        player_train = train_data[train_data['fullName'] == player]
        player_test = test_data[test_data['fullName'] == player]

        if len(player_train) < 30 or len(player_test) < 5:
            continue

        # Calculate clutch performance
        if 'plusMinusPoints' in player_train.columns:
            player_train['margin'] = abs(player_train['plusMinusPoints'])
            close_games = player_train[player_train['margin'] <= 5]

            if len(close_games) > 0:
                clutch_factor = close_games['points'].mean() / player_train['points'].mean()
            else:
                clutch_factor = 1
        else:
            clutch_factor = 1

        for _, row in player_test.iterrows():
            if pd.notna(row['rolling_5g_points']) and pd.notna(row['player_threshold']):
                # Baseline
                baseline_pred = row['rolling_5g_points']

                # Pressure adjustment (simplified - assuming pressure games)
                pressure_adjusted = baseline_pred * clutch_factor

                # Compare
                actual = row['over_threshold']
                baseline_correct = (1 if baseline_pred > row['player_threshold'] else 0) == actual
                pressure_correct = (1 if pressure_adjusted > row['player_threshold'] else 0) == actual

                if baseline_correct != pressure_correct:
                    if pressure_correct:
                        improvements.append(1)
                    else:
                        improvements.append(-1)

    if improvements:
        avg_improvement = np.mean(improvements) * 0.01
        positive_rate = sum(1 for i in improvements if i > 0) / len(improvements)

        print(f"✅ Situational Pressure Feature Results:")
        print(f"   - Average accuracy improvement: {avg_improvement:+.2%}")
        print(f"   - Positive impact rate: {positive_rate:.1%}")
        print(f"   - Recommendation: {'ADD' if avg_improvement > 0.01 else 'SKIP'}")

        return avg_improvement
    else:
        print("❌ Insufficient data for situational pressure test")
        return 0

def run_all_tests():
    """Run all feature tests and generate report."""
    print("\n" + "="*80)
    print("FEATURE MIGRATION TEST SUITE")
    print("="*80)
    print("Testing each experimental feature for potential improvement...")
    print(f"Test date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # Run individual tests
    results['seasonal'] = test_seasonal_trends_feature()
    results['peak_performance'] = test_peak_performance_feature()
    results['team_dynamics'] = test_team_dynamics_feature()
    results['situational_pressure'] = test_situational_pressure_feature()

    # Generate summary
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)

    recommended_features = []

    for feature, improvement in results.items():
        if improvement > 0.01:  # 1% improvement threshold
            status = "✅ RECOMMENDED"
            recommended_features.append(feature)
        elif improvement > 0:
            status = "⚠️  MARGINAL"
        else:
            status = "❌ NOT RECOMMENDED"

        print(f"{feature:25}: {improvement:+.2%} improvement | {status}")

    # Final recommendation
    print("\n" + "="*80)
    print("MIGRATION PLAN")
    print("="*80)

    if recommended_features:
        print(f"\n🎯 Features to migrate to final_predictions_system.py:")
        for feature in recommended_features:
            print(f"   • {feature}")

        print(f"\nExpected cumulative improvement: ~{sum(results[f] for f in recommended_features):.2%}")
    else:
        print("\n⚠️  No features show significant improvement (>1%).")
        print("   Consider adjusting thresholds or feature engineering.")

    return results, recommended_features

if __name__ == "__main__":
    run_all_tests()