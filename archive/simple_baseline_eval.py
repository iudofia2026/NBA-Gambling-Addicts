"""
Simplified NBA Baseline Evaluation

Quick baseline evaluation that works with our current data structure.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_evaluate_baselines():
    """Load data and evaluate simple baselines."""

    print("=== SIMPLIFIED NBA BASELINE EVALUATION ===")

    # Load engineered features
    data = pd.read_csv('../data/processed/engineered_features.csv')

    print(f"✓ Loaded {len(data):,} games with {data.shape[1]} features")
    print(f"✓ Players: {data['fullName'].nunique()}")
    print(f"✓ Over rate: {data['over_threshold'].mean():.1%}")
    print()

    # Use 80/20 random split for quick evaluation
    np.random.seed(42)
    train_mask = np.random.random(len(data)) < 0.8

    train_data = data[train_mask].copy()
    test_data = data[~train_mask].copy()

    print(f"Train: {len(train_data):,} games | Test: {len(test_data):,} games")
    print(f"Train over rate: {train_data['over_threshold'].mean():.1%}")
    print(f"Test over rate: {test_data['over_threshold'].mean():.1%}")
    print()

    # Baseline 1: Season Average vs Threshold
    print("1. SEASON AVERAGE BASELINE")
    player_avg = train_data.groupby('fullName')['points'].mean()
    test_data['season_avg_pred'] = test_data['fullName'].map(player_avg)
    test_data['season_baseline'] = (test_data['season_avg_pred'] > test_data['player_threshold']).astype(int)

    season_accuracy = (test_data['season_baseline'] == test_data['over_threshold']).mean()
    print(f"   Accuracy: {season_accuracy:.3f}")

    # Baseline 2: 5-Game Rolling Average vs Threshold
    print("2. ROLLING 5-GAME BASELINE")
    test_with_rolling = test_data[test_data['rolling_5g_points'].notna()].copy()
    if len(test_with_rolling) > 0:
        rolling_pred = (test_with_rolling['rolling_5g_points'] > test_with_rolling['player_threshold']).astype(int)
        rolling_accuracy = (rolling_pred == test_with_rolling['over_threshold']).mean()
        print(f"   Accuracy: {rolling_accuracy:.3f} ({len(test_with_rolling):,} games)")
    else:
        rolling_accuracy = 0
        print("   No rolling data available")

    # Baseline 3: Random
    print("3. RANDOM BASELINE")
    np.random.seed(42)
    random_pred = np.random.randint(0, 2, size=len(test_data))
    random_accuracy = (random_pred == test_data['over_threshold']).mean()
    print(f"   Accuracy: {random_accuracy:.3f}")
    print()

    # Player-specific analysis
    print("PLAYER-SPECIFIC RESULTS (Season Average Baseline):")
    print("-" * 60)

    player_results = []
    for player in test_data['fullName'].unique():
        player_test = test_data[test_data['fullName'] == player]
        if len(player_test) > 0:
            player_acc = (player_test['season_baseline'] == player_test['over_threshold']).mean()
            threshold = player_test['player_threshold'].iloc[0]
            avg_pred = player_test['season_avg_pred'].iloc[0] if not player_test['season_avg_pred'].isna().all() else 0

            player_results.append({
                'player': player,
                'test_games': len(player_test),
                'accuracy': player_acc,
                'threshold': threshold,
                'pred_avg': avg_pred,
                'actual_over_rate': player_test['over_threshold'].mean()
            })

    player_df = pd.DataFrame(player_results).sort_values('accuracy', ascending=False)

    for _, row in player_df.iterrows():
        print(f"{row['player']:20s} | {row['test_games']:3d} games | {row['accuracy']:.3f} acc | "
              f"thresh: {row['threshold']:4.1f} | pred: {row['pred_avg']:4.1f}")

    print()

    # Summary
    results = {
        'season_average_accuracy': season_accuracy,
        'rolling_5g_accuracy': rolling_accuracy,
        'random_accuracy': random_accuracy,
        'best_baseline': 'Season Average' if season_accuracy >= rolling_accuracy else 'Rolling 5-Game',
        'beat_random': max(season_accuracy, rolling_accuracy) - random_accuracy,
        'avg_player_accuracy': player_df['accuracy'].mean(),
        'best_player_accuracy': player_df['accuracy'].max(),
        'worst_player_accuracy': player_df['accuracy'].min()
    }

    # Save results
    os.makedirs('../data/processed', exist_ok=True)
    results_df = pd.DataFrame([results])
    results_df.to_csv('../data/processed/baseline_results.csv', index=False)
    player_df.to_csv('../data/processed/baseline_player_results.csv', index=False)

    print("SUMMARY:")
    print(f"✅ Best baseline: {results['best_baseline']}")
    print(f"✅ Best accuracy: {max(season_accuracy, rolling_accuracy):.3f}")
    print(f"✅ Beat random by: +{results['beat_random']:.3f}")
    print(f"✅ Player accuracy range: {results['worst_player_accuracy']:.3f} - {results['best_player_accuracy']:.3f}")
    print(f"✅ Average player accuracy: {results['avg_player_accuracy']:.3f}")

    return results, player_df

if __name__ == "__main__":
    results, player_results = load_and_evaluate_baselines()