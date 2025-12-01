#!/usr/bin/env python3
"""
Setup Verification Script
Tests that all components are properly configured
"""

import sys
import os

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")

    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'requests': 'requests',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'joblib': 'joblib'
    }

    failed = []

    for module, package_name in required_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - NOT INSTALLED")
            failed.append(package_name)

    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print(f"   Install with: pip install {' '.join(failed)}")
        return False

    print("\n✅ All packages installed successfully")
    return True


def test_api_key():
    """Test that API key is configured."""
    print("\nTesting API key configuration...")

    # Try loading from .env file first
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("  ✓ python-dotenv available")
    except ImportError:
        print("  ⚠ python-dotenv not installed (optional)")

    api_key = os.getenv('ODDS_API_KEY')

    if not api_key:
        print("  ✗ ODDS_API_KEY not found")
        print("\n❌ API key not configured")
        print("   1. Get a key from https://the-odds-api.com/")
        print("   2. Add to .env file: ODDS_API_KEY='your_key'")
        print("   3. Or export: export ODDS_API_KEY='your_key'")
        return False

    if api_key == 'your_api_key_here':
        print("  ✗ API key is placeholder value")
        print("\n❌ Please set your actual API key in .env")
        return False

    print(f"  ✓ API key configured (length: {len(api_key)})")
    print("\n✅ API key is set")
    return True


def test_directories():
    """Test that required directories exist."""
    print("\nTesting directory structure...")

    required_dirs = [
        'data/raw',
        'data/processed',
        'models',
        'src',
        'results'
    ]

    missing = []

    for directory in required_dirs:
        if os.path.isdir(directory):
            print(f"  ✓ {directory}/")
        else:
            print(f"  ✗ {directory}/ - MISSING")
            missing.append(directory)

    if missing:
        print(f"\n⚠ Missing directories: {', '.join(missing)}")
        print(f"   Create with: mkdir -p {' '.join(missing)}")
        return False

    print("\n✅ All directories present")
    return True


def test_models():
    """Test that trained models exist."""
    print("\nTesting for trained models...")

    model_files = [
        'models/logistic_regression_model.pkl',
        'models/random_forest_model.pkl',
        'models/xgboost_model.pkl',
        'models/feature_columns.pkl'
    ]

    found = 0
    missing = []

    for model_file in model_files:
        if os.path.isfile(model_file):
            print(f"  ✓ {model_file}")
            found += 1
        else:
            print(f"  ✗ {model_file} - MISSING")
            missing.append(model_file)

    if found == 0:
        print("\n⚠ No trained models found")
        print("   Train models by running: python src/final_ml_models.py")
        return False
    elif missing:
        print(f"\n⚠ Some models missing ({found}/{len(model_files)})")
        print("   This is OK, but you may want to train all models")
        return True

    print(f"\n✅ All {found} models found")
    return True


def test_data():
    """Test that processed data exists."""
    print("\nTesting for processed data...")

    data_file = 'data/processed/engineered_features.csv'

    if os.path.isfile(data_file):
        import pandas as pd
        df = pd.read_csv(data_file)
        print(f"  ✓ {data_file}")
        print(f"  ✓ {len(df):,} historical records")
        print(f"  ✓ {df['fullName'].nunique()} unique players")
        print("\n✅ Historical data ready")
        return True
    else:
        print(f"  ✗ {data_file} - MISSING")
        print("\n⚠ No processed data found")
        print("   Process data by running:")
        print("     python src/data_cleaning.py")
        print("     python src/feature_engineering.py")
        return False


def test_api_connection():
    """Test connection to The Odds API."""
    print("\nTesting API connection...")

    api_key = os.getenv('ODDS_API_KEY')

    if not api_key or api_key == 'your_api_key_here':
        print("  ⚠ Skipping (no valid API key)")
        return None

    try:
        import requests

        url = "https://api.the-odds-api.com/v4/sports"
        params = {'apiKey': api_key}

        print("  → Making test request...")
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            remaining = response.headers.get('x-requests-remaining', 'unknown')
            print(f"  ✓ API connection successful")
            print(f"  ✓ Requests remaining: {remaining}")
            print("\n✅ API is working")
            return True
        elif response.status_code == 401:
            print(f"  ✗ Invalid API key (401 Unauthorized)")
            print("\n❌ API key is invalid")
            print("   Check your key at https://the-odds-api.com/account/")
            return False
        else:
            print(f"  ✗ API returned status {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            print("\n⚠ API connection issue")
            return False

    except requests.exceptions.Timeout:
        print("  ✗ Request timed out")
        print("\n⚠ Check your internet connection")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print("\n⚠ API test failed")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("NBA Betting Predictions System - Setup Verification")
    print("=" * 60)
    print()

    results = {
        'Imports': test_imports(),
        'API Key': test_api_key(),
        'Directories': test_directories(),
        'Models': test_models(),
        'Data': test_data(),
        'API Connection': test_api_connection()
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"

        print(f"{test_name:20s} {status}")

    critical_tests = ['Imports', 'API Key', 'Directories']
    critical_passed = all(results[test] for test in critical_tests if test in results)

    print("\n" + "=" * 60)

    if critical_passed and all(v in [True, None] for v in results.values()):
        print("🎉 SETUP COMPLETE - You're ready to run predictions!")
        print("\nNext step:")
        print("  python src/daily_predictions.py")
        return 0
    elif critical_passed:
        print("⚠️  SETUP MOSTLY COMPLETE - Some optional components missing")
        print("\nYou can still run predictions, but may need to train models first.")
        return 0
    else:
        print("❌ SETUP INCOMPLETE - Please fix the errors above")
        print("\nSee SETUP.md for detailed instructions")
        return 1


if __name__ == "__main__":
    sys.exit(main())
