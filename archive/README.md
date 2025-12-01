# NBA Betting System - Archive Directory

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
