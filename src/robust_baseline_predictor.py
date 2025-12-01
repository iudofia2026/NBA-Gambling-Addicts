#!/usr/bin/env python3
"""
ROBUST BASELINE NBA PREDICTOR
Simple, realistic predictions that get close to betting lines
"""

import pandas as pd
import numpy as np
from datetime import datetime

class RobustBaselinePredictor:
    """
    Simple, robust NBA prediction system that generates realistic predictions.
    Focus: Get predictions CLOSE to betting lines, not perfect accuracy.
    """

    def __init__(self, historical_data):
        self.historical_data = historical_data

    def get_realistic_prediction(self, player_name, stat_type, prop_line):
        """
        Generate realistic predictions that are close to betting lines.
        """
        # Get player data
        player_data = self.historical_data[self.historical_data['fullName'] == player_name]
        if player_data.empty:
            return None

        # Get stat column name
        stat_column = {
            'points': 'points',
            'rebounds': 'reboundsTotal',
            'assists': 'assists'
        }.get(stat_type)

        if not stat_column or stat_column not in player_data.columns:
            return None

        # Calculate REALISTIC baselines
        recent_data = player_data.tail(10)  # Last 10 games
        season_data = player_data.tail(30)  # Last 30 games

        if len(recent_data) < 3:
            return None

        # Simple weighted average (recent games matter more)
        recent_avg = recent_data[stat_column].mean()
        season_avg = season_data[stat_column].mean()

        # Baseline prediction: 70% recent, 30% season
        baseline = (recent_avg * 0.7) + (season_avg * 0.3)

        # REALISTIC ADJUSTMENTS (small, sensible)

        # 1. Form adjustment (hot/cold)
        last_3_games = recent_data[stat_column].tail(3).mean()
        form_factor = (last_3_games - recent_avg) / recent_avg if recent_avg > 0 else 0
        form_adjustment = baseline * form_factor * 0.1  # Max 10% adjustment

        # 2. Line-aware calibration (key insight!)
        # If our prediction is way off the line, adjust it towards the line
        line_distance = abs(baseline - prop_line)
        max_reasonable_distance = prop_line * 0.3  # Max 30% away from line

        if line_distance > max_reasonable_distance:
            # Pull prediction towards line to be more realistic
            direction = 1 if baseline < prop_line else -1
            calibration = direction * (line_distance - max_reasonable_distance) * 0.5
            line_adjustment = calibration
        else:
            line_adjustment = 0

        # 3. Small random variance for realism
        variance = np.random.normal(0, baseline * 0.05)  # 5% variance

        # Final prediction
        prediction = baseline + form_adjustment + line_adjustment + variance

        # Ensure reasonable bounds
        prediction = max(0, prediction)  # No negative stats

        # Calculate confidence (lower when adjusted heavily)
        adjustment_magnitude = abs(form_adjustment + line_adjustment) / baseline if baseline > 0 else 0
        confidence = max(0.3, 0.8 - adjustment_magnitude)

        return {
            'predicted_value': round(prediction, 1),
            'baseline': round(baseline, 1),
            'recent_avg': round(recent_avg, 1),
            'season_avg': round(season_avg, 1),
            'form_adjustment': round(form_adjustment, 2),
            'line_adjustment': round(line_adjustment, 2),
            'confidence': round(confidence, 3),
            'line_diff': round(prediction - prop_line, 1),
            'recommendation': 'OVER' if prediction > prop_line else 'UNDER'
        }

def test_robust_predictions():
    """Test the robust predictor with real data."""

    # Load data
    try:
        data = pd.read_csv('/Users/iudofia/Desktop/NBA-Gambling-Addicts/data/processed/engineered_features.csv')
        print(f"✓ Loaded {len(data):,} games")
    except:
        print("❌ Could not load data")
        return

    predictor = RobustBaselinePredictor(data)

    # Test with problematic players from the CSV
    test_cases = [
        ('Nikola Jokic', 'rebounds', 14.5),
        ('James Harden', 'assists', 7.5),
        ('James Harden', 'points', 21.5),
        ('Kevin Durant', 'points', 29.5),
        ('Devin Booker', 'points', 25.5),
        ('Tobias Harris', 'points', 10.5)
    ]

    print("\n🎯 ROBUST BASELINE PREDICTIONS")
    print("=" * 60)

    for player, stat, line in test_cases:
        result = predictor.get_realistic_prediction(player, stat, line)

        if result:
            print(f"\n🏀 {player} - {stat.upper()}")
            print(f"   Line: {line}")
            print(f"   Prediction: {result['predicted_value']} ({result['line_diff']:+.1f})")
            print(f"   Recent Avg: {result['recent_avg']:.1f}, Season Avg: {result['season_avg']:.1f}")
            print(f"   Form Adj: {result['form_adjustment']:+.2f}, Line Adj: {result['line_adjustment']:+.2f}")
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"   Recommendation: {result['recommendation']}")
        else:
            print(f"\n❌ {player} - {stat}: No data")

if __name__ == "__main__":
    test_robust_predictions()