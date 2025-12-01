"""
Update tests and execute cleanup
Handles test updates and removes unnecessary files
"""

import os
import re
from pathlib import Path

def update_test_references():
    """Update test references to use the new optimized prediction system."""

    print("\n" + "=" * 60)
    print("UPDATING TEST REFERENCES")
    print("=" * 60)

    # Files to update
    test_updates = {
        'tests/unit/test_daily_predictions.py': {
            'old_import': 'from daily_predictions import DailyPredictor',
            'new_import': 'from final_predictions_optimized import OptimizedNBAPredictionsSystem as DailyPredictor',
            'old_patch': 'daily_predictions.NBAOddsClient',
            'new_patch': 'final_predictions_optimized.NBAOddsClient'
        }
    }

    for test_file, updates in test_updates.items():
        if Path(test_file).exists():
            print(f"\n📝 Updating {test_file}...")

            with open(test_file, 'r') as f:
                content = f.read()

            # Update imports
            content = content.replace(updates['old_import'], updates['new_import'])

            # Update patch references
            content = content.replace(updates['old_patch'], updates['new_patch'])

            # Write back
            with open(test_file, 'w') as f:
                f.write(content)

            print(f"  ✓ Updated imports and patches")
        else:
            print(f"\n⚠️  Test file not found: {test_file}")

def check_integration_tests():
    """Check if integration tests need updates."""

    print("\n" + "=" * 60)
    print("CHECKING INTEGRATION TESTS")
    print("=" * 60)

    integration_tests = list(Path('tests/integration').glob('*.py'))

    for test_file in integration_tests:
        print(f"\n🔍 Checking {test_file.name}...")

        with open(test_file, 'r') as f:
            content = f.read()

        # Look for references to old prediction systems
        if 'daily_predictions' in content:
            print(f"  ⚠️  Contains references to daily_predictions")
        if 'final_predictions_system' in content:
            print(f"  ⚠️  Contains references to final_predictions_system")
        if 'enhanced_predictions' in content:
            print(f"  ⚠️  Contains references to enhanced_predictions")
        if 'final_predictions_optimized' in content:
            print(f"  ✓ Already references optimized system")

def create_migration_readme():
    """Create a README in the archive directory explaining the migration."""

    print("\n" + "=" * 60)
    print("CREATING MIGRATION DOCUMENTATION")
    print("=" * 60)

    readme_content = """# NBA Betting System - Archive Directory

This directory contains experimental and deprecated code from the development of the NBA betting prediction system.

## Migration Summary

After extensive testing, the repository has been simplified to use only the best-performing prediction system:

### Final Performance Results (on 2,690 test games):
- **`final_predictions_optimized.py`**: 60.41% accuracy ⭐ (PRODUCTION SYSTEM)
- `final_predictions_system.py`: 60.30% accuracy (archived)
- `final_predictions_enhanced.py`: 60.26% accuracy (archived)

### Directory Structure

#### `experimental_predictions/`
- `final_predictions_system.py` - Original 3-iteration system
- `final_predictions_enhanced.py` - Attempted 9-iteration migration (worse performance)
- `enhanced_predictions_v*.py` - Iterative development versions

#### `feature_experiments/`
- `advanced_features_v1.py` - Early feature experiments
- `external_factors_v1.py` - External factor features (travel, B2B)
- `shot_quality_v2.py` - Shot quality analysis features
- `evidence_features_v4.py` - Evidence-based features (home-court, rest)
- `matchup_features.py` - Basic matchup analysis
- `matchup_analytics_v5.py` - Advanced matchup analytics
- `advanced_analytics_v6.py` - Complete 9-iteration experimental system

#### `deprecated_models/`
- `ml_models_simple.py` - Simplified ML implementations
- `baseline_models.py` - Baseline model comparisons

#### `analysis_tools/`
- `simple_age_analysis.py` - Age-based performance analysis
- `age_analysis.py` - Detailed age analysis

#### `testing_framework/`
- `feature_tester.py` - Framework for testing feature additions
- `test_feature_migration.py` - Specific feature migration tests
- `simulate_feature_impact.py` - Feature impact simulation
- `compare_prediction_systems.py` - System comparison tool

## Key Learnings

1. **More iterations ≠ better performance**: The 9-iteration system performed worse than the original 3-iteration system.

2. **Conservative feature selection works best**: Small, conservative adjustments performed better than large, complex feature sets.

3. **Overfitting risk**: Adding too many features without sufficient validation leads to worse performance.

## Current Production System

The production system (`../src/final_predictions_optimized.py`) includes:
- Conservative feature additions from the experimental work
- Optimized weights based on testing
- Enhanced form analysis with volatility metrics
- Improved matchup analysis with sample size validation
- Maintained the best accuracy while adding insights

## Date Archived
2025-12-01
"""

    with open('archive/README.md', 'w') as f:
        f.write(readme_content)

    print("✓ Created archive/README.md with migration documentation")

def update_main_readme():
    """Update the main README to reflect the simplified structure."""

    print("\n" + "=" * 60)
    print("UPDATING MAIN README")
    print("=" * 60)

    if Path('README.md').exists():
        with open('README.md', 'r') as f:
            content = f.read()

        # Update the prediction systems section
        old_section = """- **Prediction systems** (`src/`)
  - `daily_predictions.py`: Primary, model-based daily prediction script using the persisted ensemble and live odds.
  - `final_predictions_system.py`: "Final" prediction engine that layers momentum, team chemistry, and matchup history on top of baselines.
  - `advanced_analytics_v6.py`: Advanced feature generators and `CompleteNBAPredictor` that combines evidence-based, matchup, and advanced seasonal/pressure signals.
  - Additional experimental/iterative scripts (`enhanced_predictions_*`, `matchup_analytics_v5.py`, `evidence_features_v4.py`, etc.)."""

        new_section = """- **Prediction system** (`src/`)
  - `final_predictions_optimized.py`: ⭐ **PRODUCTION SYSTEM** - Best performing model (60.41% accuracy) with conservative feature additions from experimental testing."""

        content = content.replace(old_section, new_section)

        # Update usage section
        old_usage = """```bash
source venv/bin/activate
python src/daily_predictions.py
```"""

        new_usage = """```bash
source venv/bin/activate
python src/final_predictions_optimized.py
```"""

        content = content.replace(old_usage, new_usage)

        # Also update the alternative option
        content = content.replace(
            "python src/final_predictions_system.py",
            "python src/final_predictions_optimized.py"
        )

        with open('README.md', 'w') as f:
            f.write(content)

        print("✓ Updated README.md with simplified structure")
    else:
        print("⚠️  README.md not found")

def main():
    """Execute all updates."""

    print("\n" + "=" * 80)
    print("NBA BETTING SYSTEM - TEST UPDATES & CLEANUP")
    print("=" * 80)

    # Update tests
    update_test_references()

    # Check integration tests
    check_integration_tests()

    # Create migration documentation
    create_migration_readme()

    # Update main README
    update_main_readme()

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Review the updates above")
    print("2. Run: python3 cleanup_repo.py --execute")
    print("3. Test the optimized system to ensure it works:")
    print("   python3 src/final_predictions_optimized.py")
    print("\n⚠️  Make sure you have ODDS_API_KEY set before testing!")

if __name__ == "__main__":
    main()