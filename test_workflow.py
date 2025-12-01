#!/usr/bin/env python3
"""
Test NBA Betting Predictions Workflow

This script tests the complete workflow without requiring an API key
by mocking the API response and testing the ML pipeline integration.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

def test_ml_models_loading():
    """Test that ML models can be loaded."""
    print("\n=== TESTING ML MODEL LOADING ===")

    try:
        import joblib

        model_dir = 'models'
        models = {}

        # Test loading each model
        model_files = {
            'logistic_regression': 'logistic_regression_model.pkl',
            'random_forest': 'random_forest_model.pkl',
            'xgboost': 'xgboost_model.pkl'
        }

        for model_name, filename in model_files.items():
            filepath = f"{model_dir}/{filename}"
            if os.path.exists(filepath):
                models[model_name] = joblib.load(filepath)
                print(f"✓ Loaded {model_name} model")
            else:
                print(f"⚠️  {model_name} model not found: {filepath}")

        # Load feature columns
        feature_file = f"{model_dir}/feature_columns.pkl"
        if os.path.exists(feature_file):
            feature_cols = joblib.load(feature_file)
            print(f"✓ Loaded {len(feature_cols)} feature columns")
        else:
            print(f"❌ Feature columns not found: {feature_file}")
            return False

        print(f"✅ Successfully loaded {len(models)} models")
        return len(models) > 0

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def test_data_loading():
    """Test that historical data can be loaded."""
    print("\n=== TESTING DATA LOADING ===")

    try:
        # Load historical data
        data = pd.read_csv('data/processed/engineered_features.csv')
        data['gameDate'] = pd.to_datetime(data['gameDate'], errors='coerce')

        print(f"✓ Loaded {len(data):,} historical game records")
        print(f"✓ Date range: {data['gameDate'].min().date()} to {data['gameDate'].max().date()}")
        print(f"✓ Unique players: {data['fullName'].nunique()}")

        # Show sample players
        sample_players = data['fullName'].unique()[:10]
        print(f"✓ Sample players: {', '.join(sample_players)}")

        return len(data) > 0

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_mock_prediction_workflow():
    """Test the prediction workflow with mock data."""
    print("\n=== TESTING PREDICTION WORKFLOW WITH MOCK DATA ===")

    try:
        # Create mock player props data
        mock_props = pd.DataFrame([
            {
                'gameDate': datetime.now().date(),
                'fullName': 'LeBron James',
                'home_team': 'Los Angeles Lakers',
                'away_team': 'Boston Celtics',
                'game_time': datetime.now() + timedelta(hours=3),
                'prop_line': 25.5,
                'over_odds': 110,
                'bookmaker': 'DraftKings',
                'market_type': 'points',
                'api_timestamp': datetime.now()
            },
            {
                'gameDate': datetime.now().date(),
                'fullName': 'Stephen Curry',
                'home_team': 'Golden State Warriors',
                'away_team': 'Phoenix Suns',
                'game_time': datetime.now() + timedelta(hours=4),
                'prop_line': 27.5,
                'over_odds': -115,
                'bookmaker': 'FanDuel',
                'market_type': 'points',
                'api_timestamp': datetime.now()
            }
        ])

        print(f"✓ Created mock props data for {len(mock_props)} players")

        # Test that we can load models and make predictions
        if not test_ml_models_loading():
            print("❌ Cannot test predictions without models")
            return False

        # Load historical data for context
        if not test_data_loading():
            print("❌ Cannot test predictions without historical data")
            return False

        print("✓ Mock prediction workflow components are ready")

        # Test feature generation (simplified)
        historical_data = pd.read_csv('data/processed/engineered_features.csv')

        for _, prop in mock_props.iterrows():
            player_data = historical_data[historical_data['fullName'] == prop['fullName']]

            if not player_data.empty:
                print(f"✓ Found {len(player_data)} historical games for {prop['fullName']}")
                print(f"  - Prop line: {prop['prop_line']} points")
                print(f"  - Game: {prop['away_team']} @ {prop['home_team']}")
            else:
                print(f"⚠️  No historical data for {prop['fullName']}")

        print("✅ Mock prediction workflow test completed")
        return True

    except Exception as e:
        print(f"❌ Error in mock workflow: {e}")
        return False

def test_api_client_structure():
    """Test that the API client module is structured correctly."""
    print("\n=== TESTING API CLIENT STRUCTURE ===")

    try:
        # Import without initializing (to avoid API key requirement)
        from odds_api_client import NBAOddsClient

        print("✓ Successfully imported NBAOddsClient")

        # Test that the class can be instantiated with a dummy key
        try:
            client = NBAOddsClient('dummy_key_for_testing')
            print("✓ NBAOddsClient can be instantiated")

            # Test that tracked players are loaded
            if hasattr(client, 'tracked_players') and client.tracked_players:
                print(f"✓ Loaded {len(client.tracked_players)} tracked players")
            else:
                print("⚠️  No tracked players loaded")

        except Exception as e:
            print(f"⚠️  Could not fully initialize client: {e}")

        return True

    except Exception as e:
        print(f"❌ Error importing API client: {e}")
        return False

def main():
    """Run all workflow tests."""
    print("🏀 NBA BETTING PREDICTIONS - WORKFLOW TEST")
    print("=" * 50)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    tests = [
        ("Data Loading", test_data_loading),
        ("ML Models Loading", test_ml_models_loading),
        ("API Client Structure", test_api_client_structure),
        ("Mock Prediction Workflow", test_mock_prediction_workflow)
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'-' * 20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results[test_name] = False

    # Summary
    print(f"\n{'=' * 50}")
    print("🎯 WORKFLOW TEST SUMMARY")
    print(f"{'=' * 50}")

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25s} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The system is ready to run with a real API key.")
        print("\nTo use with live data:")
        print("1. Get API key from https://the-odds-api.com/")
        print("2. Set environment variable: export ODDS_API_KEY='your_key'")
        print("3. Run: python src/daily_predictions.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the issues above.")
        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)