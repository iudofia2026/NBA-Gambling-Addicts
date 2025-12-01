"""
Repository Cleanup Script
Identifies files to keep, archive, or remove based on the lazy-architect's recommendations
"""

import os
import shutil
from pathlib import Path

def create_cleanup_plan():
    """Create a comprehensive cleanup plan for the NBA repository."""

    print("=" * 80)
    print("NBA GAMBLING ADDICTS - REPOSITORY CLEANUP PLAN")
    print("=" * 80)

    # Files to KEEP in src/ (core production code)
    keep_files = {
        # Core data pipeline
        'data_cleaning.py': 'Essential for raw data processing',
        'feature_engineering.py': 'Core feature generation for ML pipeline',

        # ML models and training
        'final_ml_models.py': 'Primary model training script',
        'ml_models.py': 'Core ML utilities and helper functions',
        'scaled_lr.py': 'Required dependency for ML models',

        # Prediction systems - ONLY THE BEST
        'final_predictions_optimized.py': '⭐ BEST PERFORMING SYSTEM (60.41% accuracy)',

        # API integration
        'odds_api_client.py': 'Essential for fetching live betting odds',

        # Core utilities
        '__init__.py': 'Python package marker',
    }

    # Files to ARCHIVE (experimental/deprecated)
    archive_files = {
        # Original/alternative prediction systems
        'final_predictions_system.py': 'Original system (60.30% accuracy) - surpassed by optimized',
        'final_predictions_enhanced.py': 'Full migration attempt (60.26% accuracy) - worse performance',

        # Iterative development versions (v1-v5)
        'enhanced_predictions_v1.py': 'Development version - superseded',
        'enhanced_predictions_v2.py': 'Development version - superseded',
        'enhanced_predictions_v4.py': 'Development version - superseded',
        'enhanced_predictions_v5.py': 'Development version - superseded',
        'enhanced_daily_predictions.py': 'Alternative daily predictions - not used',

        # Feature iterations (experimental)
        'advanced_features_v1.py': 'Experimental features - integrated elsewhere',
        'external_factors_v1.py': 'Experimental features - integrated elsewhere',
        'shot_quality_v2.py': 'Experimental features - integrated elsewhere',
        'evidence_features_v4.py': 'Experimental features - integrated elsewhere',
        'matchup_features.py': 'Experimental features - integrated elsewhere',
        'matchup_analytics_v5.py': 'Experimental features - integrated elsewhere',
        'advanced_analytics_v6.py': 'Experimental 9-iteration system - not performant',

        # Alternative implementations
        'ml_models_simple.py': 'Simplified ML models - not used in production',
        'baseline_models.py': 'Baseline models - superseded by final_ml_models.py',

        # Analysis/interpretation (not core to pipeline)
        'model_interpretation.py': 'Analysis tool - not required for predictions',
        'simple_baseline_eval.py': 'Evaluation script - one-time use',
        'simple_age_analysis.py': 'Analysis script - not part of core pipeline',
        'age_analysis.py': 'Analysis script - not part of core pipeline',
    }

    # Files to REMOVE (testing/evaluation scripts)
    remove_files = {
        # Testing framework created for feature evaluation
        'feature_tester.py': 'One-time feature evaluation - no longer needed',
        'test_feature_migration.py': 'One-time feature evaluation - no longer needed',
        'simulate_feature_impact.py': 'One-time feature evaluation - no longer needed',
        'compare_prediction_systems.py': 'One-time comparison - no longer needed',

        # Test files in root
        '../test_setup.py': 'Test setup - should be in tests/',
        '../test_workflow.py': 'Test workflow - should be in tests/',
    }

    # Create archive directory
    archive_dir = Path('archive')
    archive_dir.mkdir(exist_ok=True)

    # Create subdirectories for organization
    (archive_dir / 'experimental_predictions').mkdir(exist_ok=True)
    (archive_dir / 'feature_experiments').mkdir(exist_ok=True)
    (archive_dir / 'deprecated_models').mkdir(exist_ok=True)
    (archive_dir / 'analysis_tools').mkdir(exist_ok=True)
    (archive_dir / 'testing_framework').mkdir(exist_ok=True)

    print("\n📂 CLEANUP PLAN SUMMARY:")
    print(f"\n✅ KEEP ({len(keep_files)} files):")
    for f, reason in keep_files.items():
        print(f"   • {f:<35} - {reason}")

    print(f"\n📦 ARCHIVE ({len(archive_files)} files):")
    print("   (Moving to archive/ subdirectories)")

    # Group archive files by type
    predictions = [f for f in archive_files if 'predictions' in f]
    features = [f for f in archive_files if any(x in f for x in ['features', 'analytics', 'factors'])]
    models = [f for f in archive_files if 'models' in f]
    analysis = [f for f in archive_files if 'analysis' in f]

    if predictions:
        print(f"\n   Experimental Predictions → archive/experimental_predictions/:")
        for f in predictions:
            print(f"      • {f}")
    if features:
        print(f"\n   Feature Experiments → archive/feature_experiments/:")
        for f in features:
            print(f"      • {f}")
    if models:
        print(f"\n   Deprecated Models → archive/deprecated_models/:")
        for f in models:
            print(f"      • {f}")
    if analysis:
        print(f"\n   Analysis Tools → archive/analysis_tools/:")
        for f in analysis:
            print(f"      • {f}")

    print(f"\n🗑️  REMOVE ({len(remove_files)} files):")
    for f, reason in remove_files.items():
        print(f"   • {f:<35} - {reason}")

    print("\n" + "=" * 80)
    print("EXPECTED FINAL STRUCTURE:")
    print("=" * 80)
    print("""
src/
├── __init__.py                     # Package marker
├── data_cleaning.py               # Core: Raw data processing
├── feature_engineering.py         # Core: Feature generation
├── final_ml_models.py             # Core: Model training
├── ml_models.py                   # Core: ML utilities
├── scaled_lr.py                   # Core: ML dependency
├── odds_api_client.py             # Core: API integration
└── final_predictions_optimized.py # ⭐ PRODUCTION SYSTEM

archive/                           # Preserved but out of the way
├── experimental_predictions/       # All prediction experiments
├── feature_experiments/           # All feature engineering attempts
├── deprecated_models/             # Old model implementations
├── analysis_tools/                # Analysis and evaluation scripts
└── testing_framework/             # Feature evaluation tools

tests/                             # Unchanged - all tests remain
├── unit/
├── integration/
└── validation/

data/                             # Unchanged
├── raw/
├── processed/
└── external/

models/                           # Unchanged - trained model artifacts
""")

    return keep_files, archive_files, remove_files, archive_dir

def execute_cleanup(dry_run=True):
    """Execute the cleanup plan."""
    keep_files, archive_files, remove_files, archive_dir = create_cleanup_plan()

    if dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN MODE - No files will be moved or deleted")
        print("=" * 80)
        print("\nTo execute the cleanup, run:")
        print("  python cleanup_repo.py --execute")
        return

    print("\n" + "=" * 80)
    print("EXECUTING CLEANUP...")
    print("=" * 80)

    # Archive files
    for file in archive_files:
        src_path = Path('src') / file
        if src_path.exists():
            if 'predictions' in file:
                dst_path = archive_dir / 'experimental_predictions' / file
            elif any(x in file for x in ['features', 'analytics', 'factors']):
                dst_path = archive_dir / 'feature_experiments' / file
            elif 'models' in file:
                dst_path = archive_dir / 'deprecated_models' / file
            elif 'analysis' in file:
                dst_path = archive_dir / 'analysis_tools' / file
            else:
                dst_path = archive_dir / file

            shutil.move(str(src_path), str(dst_path))
            print(f"✓ Archived: {file} → archive/{dst_path.parent.name}/{dst_path.name}")

    # Remove files
    for file in remove_files:
        # Remove the ../ prefix for paths
        clean_file = file.replace('../', '')
        if clean_file.startswith('src/'):
            file_path = Path(clean_file)
        else:
            file_path = Path('src') / clean_file if not clean_file.startswith('src') else Path(clean_file)

        if file_path.exists():
            file_path.unlink()
            print(f"✗ Removed: {file_path}")

    print("\n✅ Cleanup complete!")
    print(f"\nArchived {len(archive_files)} files")
    print(f"Removed {len(remove_files)} files")
    print(f"Kept {len(keep_files)} core files in src/")

if __name__ == "__main__":
    import sys

    # Check for --execute flag
    execute = '--execute' in sys.argv

    # Show plan first
    create_cleanup_plan()

    if execute:
        execute_cleanup(dry_run=False)
    else:
        print("\n" + "=" * 80)
        print("To execute this cleanup plan, run: python cleanup_repo.py --execute")
        print("=" * 80)