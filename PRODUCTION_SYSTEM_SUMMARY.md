# NBA Betting System - Production Summary

## Current Production System

The repository has been cleaned and simplified to use a single, best-performing prediction system:

### **Production Model**: `src/final_predictions_optimized.py`
- **Accuracy**: 60.41% (highest among all tested systems)
- **Features**: Conservative feature additions that improve performance
- **Approach**: Selective migration of only beneficial experimental features

## Performance Comparison

| System | Accuracy | Status |
|--------|----------|---------|
| **final_predictions_optimized.py** | **60.41%** | ✅ PRODUCTION |
| final_predictions_system.py | 60.30% | 📦 Archived |
| final_predictions_enhanced.py | 60.26% | 📦 Archived |

## How to Run

```bash
# Activate virtual environment
source venv/bin/activate

# Set your API key
export ODDS_API_KEY="your_api_key_here"

# Run production predictions
python src/final_predictions_optimized.py
```

## Core Production Files

The `src/` directory now contains only essential files:

1. **`data_cleaning.py`** - Raw data processing
2. **`feature_engineering.py`** - Feature generation for ML
3. **`final_ml_models.py`** - Model training pipeline
4. **`ml_models.py`** - ML utilities and helpers
5. **`scaled_lr.py`** - Required ML dependency
6. **`odds_api_client.py`** - Fetches live betting odds
7. **`final_predictions_optimized.py`** - ⭐ PRODUCTION SYSTEM
8. **`__init__.py`** - Python package marker

## Archived Files

All experimental and deprecated code has been moved to `archive/`:

- **`archive/experimental_predictions/`** - All prediction system attempts
- **`archive/feature_experiments/`** - Feature engineering experiments
- **`archive/deprecated_models/`** - Old model implementations
- **`archive/analysis_tools/`** - Analysis and evaluation scripts

See `archive/README.md` for detailed documentation.

## Key Features of Production System

### Conservative Feature Additions
- Enhanced form analysis with volatility metrics
- Improved matchup analysis with sample size validation
- Optimized team chemistry calculations
- Better confidence scoring

### Performance Optimizations
- Limited adjustments to prevent overfitting
- Conservative weights focused on proven features
- Maintained simplicity while adding value

## Data Requirements

The system requires:
- `data/processed/engineered_features.csv` - Historical features
- `models/*.pkl` - Trained model artifacts
- Valid `ODDS_API_KEY` environment variable

## Testing

Run tests with:
```bash
pytest tests/
```

## Date of Cleanup
December 1, 2024

## Next Steps

1. Monitor production performance
2. Retrain models periodically with new data
3. Consider automated daily runs
4. Add more players to dataset for broader coverage