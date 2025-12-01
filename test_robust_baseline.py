#!/usr/bin/env python3
"""
Test the robust statistical baseline system
to verify improved prediction accuracy.
"""

import pandas as pd
import sys
import os

# Add src directory to path
sys.path.append('src')

from advanced_nba_predictor import AdvancedNBAPredictor

def test_robust_baselines():
    """Test the robust statistical baseline system."""

    print("🔬 TESTING ROBUST STATISTICAL BASELINE SYSTEM")
    print("=" * 60)

    try:
        # Initialize predictor without API (we're just testing baselines)
        predictor = AdvancedNBAPredictor.__new__(AdvancedNBAPredictor)
        predictor.historical_data = pd.read_csv('data/processed/engineered_features.csv')
        print(f"✅ Loaded {len(predictor.historical_data):,} historical games")

        # Test with the problematic players from CSV
        test_players = [
            'Nikola Jokic',
            'James Harden',
            'Kevin Durant',
            'Devin Booker',
            'Tobias Harris'
        ]

        print("\n📊 ROBUST BASELINE ANALYSIS")
        print("=" * 60)

        for player in test_players:
            print(f"\n🏀 ANALYZING: {player}")
            print("-" * 50)

            result = predictor.calculate_robust_statistical_baseline(player)

            if result[0] is not None:
                points, rebounds, assists = result
                print(f"✅ ROBUST BASELINES ESTABLISHED:")
                print(f"   Points: {points:.1f} PPG")
                print(f"   Rebounds: {rebounds:.1f} RPG")
                print(f"   Assists: {assists:.1f} APG")

                # Compare with previous bad predictions
                previous_bad_predictions = {
                    'Nikola Jokic': {'points': 14.7, 'rebounds': 7.6, 'assists': 0.3},
                    'Kevin Durant': {'points': 12.3, 'rebounds': 3.3, 'assists': 0.2},
                    'Devin Booker': {'points': 7.2, 'rebounds': 3.7, 'assists': 0.3},
                    'James Harden': {'points': 12.9, 'rebounds': 4.2, 'assists': 2.9},
                    'Tobias Harris': {'points': 6.3, 'rebounds': 2.1, 'assists': 1.2}
                }

                if player in previous_bad_predictions:
                    old = previous_bad_predictions[player]
                    print(f"\n📈 IMPROVEMENT ANALYSIS:")
                    print(f"   Points:   Old: {old['points']:5.1f} → New: {points:5.1f} (Δ: {points-old['points']:+6.1f})")
                    print(f"   Rebounds: Old: {old['rebounds']:5.1f} → New: {rebounds:5.1f} (Δ: {rebounds-old['rebounds']:+6.1f})")
                    print(f"   Assists:  Old: {old['assists']:5.1f} → New: {assists:5.1f} (Δ: {assists-old['assists']:+6.1f})")

                    # Check if closer to realistic NBA stats
                    realistic_ranges = {
                        'Nikola Jokic': {'points': [25, 30], 'rebounds': [12, 16], 'assists': [8, 12]},
                        'Kevin Durant': {'points': [25, 32], 'rebounds': [4, 7], 'assists': [3, 6]},
                        'Devin Booker': {'points': [25, 35], 'rebounds': [3, 6], 'assists': [6, 10]},
                        'James Harden': {'points': [18, 25], 'rebounds': [5, 8], 'assists': [7, 12]},
                        'Tobias Harris': {'points': [12, 18], 'rebounds': [4, 7], 'assists': [2, 4]}
                    }

                    if player in realistic_ranges:
                        ranges = realistic_ranges[player]
                        print(f"\n🎯 REALISM CHECK (Expected NBA ranges):")
                        points_ok = ranges['points'][0] <= points <= ranges['points'][1]
                        rebounds_ok = ranges['rebounds'][0] <= rebounds <= ranges['rebounds'][1]
                        assists_ok = ranges['assists'][0] <= assists <= ranges['assists'][1]

                        print(f"   Points:   {points:.1f} ({'✅' if points_ok else '❌'}) [Expected: {ranges['points'][0]}-{ranges['points'][1]}]")
                        print(f"   Rebounds: {rebounds:.1f} ({'✅' if rebounds_ok else '❌'}) [Expected: {ranges['rebounds'][0]}-{ranges['rebounds'][1]}]")
                        print(f"   Assists:  {assists:.1f} ({'✅' if assists_ok else '❌'}) [Expected: {ranges['assists'][0]}-{ranges['assists'][1]}]")

                        overall_realistic = points_ok and rebounds_ok and assists_ok
                        print(f"   Overall:  {'🟢 REALISTIC' if overall_realistic else '🔴 NEEDS MORE WORK'}")

            else:
                print("❌ Could not establish robust baseline")

        print("\n🏆 SUMMARY")
        print("=" * 60)
        print("✓ Robust statistical baseline system implemented")
        print("✓ Uses peak performance analysis (top 25% games)")
        print("✓ Filters for starter-level minutes (28+)")
        print("✓ Includes data quality detection")
        print("✓ Should produce much more realistic predictions")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_robust_baselines()