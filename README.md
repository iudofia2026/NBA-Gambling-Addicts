## NBA Player Prop Prediction System

A production-ready machine learning pipeline that predicts NBA player prop outcomes (over/under on points, rebounds, assists) using engineered features, ensemble models, and live betting odds from The Odds API.

The repository contains everything from raw data processing and feature engineering to model training, evaluation, and multiple live prediction engines, backed by an automated test suite.

### Core Capabilities

- **End-to-end ML pipeline**: Data cleaning, feature engineering, model training, evaluation, and prediction.
- **Ensemble models**: Logistic Regression (with scaling), Random Forest, and optional XGBoost, combined into an ensemble for robust predictions.
- **Live odds integration**: Fetches current NBA player props and odds from The Odds API and formats them for the ML pipeline.
- **Daily prediction scripts**: Multiple entry points to generate daily recommendations with confidence scores and CSV exports.
- **Advanced analytics**: Iterative feature sets (evidence-based features, matchup analytics, advanced momentum and chemistry features) for richer predictions.
- **Comprehensive testing**: Unit, integration, and validation tests for data, models, and API integration.

### High-Level Architecture

- **Data layer**
  - `data/processed/engineered_features.csv`: Main feature dataset used for training and predictions.
  - Additional processed artifacts (cleaning summaries, feature importance, age analysis, etc.).
- **Model artifacts** (`models/`)
  - `logistic_regression_model.pkl`, `random_forest_model.pkl`, `xgboost_model.pkl` (optional).
  - `feature_columns.pkl` (feature schema) and `label_encoders.pkl` (categorical encoders) shared between training and inference.
- **Core ML & pipeline code** (`src/`)
  - `ml_models.py`: General model training pipeline (train/validation split, feature prep, trainer classes, evaluation helpers).
  - `final_ml_models.py`: Leakage-safe training script that constrains features to pre-game signals only.
  - `data_cleaning.py`, `feature_engineering.py`: Data preparation and feature creation for `engineered_features.csv`.
- **Prediction systems** (`src/`)
  - `daily_predictions.py`: Primary, model-based daily prediction script using the persisted ensemble and live odds.
  - `final_predictions_system.py`: "Final" prediction engine that layers momentum, team chemistry, and matchup history on top of baselines.
  - `advanced_analytics_v6.py`: Advanced feature generators and `CompleteNBAPredictor` that combines evidence-based, matchup, and advanced analytics iterations.
  - Additional experimental/iterative scripts (`enhanced_predictions_*`, `matchup_analytics_v5.py`, `evidence_features_v4.py`, etc.).
- **Odds API integration**
  - `odds_api_client.py`: Handles HTTP calls to The Odds API, request throttling, and transformation into ML-ready DataFrames.

### Getting Started

#### 1. Install dependencies

```bash
./setup.sh
```

This script creates a virtual environment and installs packages from `requirements.txt`.

#### 2. Configure API access

```bash
echo "ODDS_API_KEY=your_api_key_here" > .env
```

- Obtain a free key from `https://the-odds-api.com` (free tier: 500 requests/month).
- The code reads `ODDS_API_KEY` from the environment; `.env.example` documents all relevant variables.

#### 3. Train (or retrain) models

If models are not present or you want to refresh them:

```bash
source venv/bin/activate
python src/final_ml_models.py
```

- Uses `data/processed/engineered_features.csv`.
- Removes leakage-prone columns, prepares features, trains logistic regression and random forest, and saves models plus encoders and feature schema to `models/`.

### Running Predictions

#### Option A: Standard daily predictions (ensemble models)

```bash
source venv/bin/activate
python src/final_predictions_optimized.py
```

- Loads serialized models and feature definitions from `models/`.
- Loads historical feature data from `data/processed/engineered_features.csv`.
- Fetches todays NBA player props and odds via `NBAOddsClient`.
- Generates ensemble predictions for each prop and filters to high-confidence picks (confidence > 0.6 and full model agreement).
- Prints human-readable recommendations and writes all predictions to `data/processed/daily_predictions_<timestamp>.csv`.

#### Option B: Final prediction system with richer context

```bash
source venv/bin/activate
python src/final_predictions_optimized.py
```

- Uses the same historical dataset and models but adds:
  - Momentum signals (short-term trends and streaks).
  - Team chemistry and usage share estimates.
  - Individual matchup history versus specific opponents.
- Produces context-rich recommendations with descriptive insights (form, chemistry, matchup history) and saves outputs under `data/processed/final_predictions_<timestamp>.csv`.

#### Option C: Complete 9-iteration analytics (advanced experimentation)

```bash
source venv/bin/activate
python src/advanced_analytics_v6.py
```

- Uses `CompleteNBAPredictor` to combine:
  - Evidence-based features (e.g., home-court advantage, rest).
  - Matchup analytics (player vs team, role, usage).
  - Advanced seasonal/peak/pressure signals.
- Intended for research/experimentation; outputs `data/processed/ultimate_predictions_<timestamp>.csv`.

### Testing and Validation

- **Unit tests** (`tests/unit/`)
  - `test_ml_models.py`: Data loading, feature preparation, trainer behavior, evaluation helpers, scaling wrapper (`ScaledLogisticRegression`).
  - `test_daily_predictions.py`: Initialization, model loading, feature generation, prediction logic, display formatting, and edge cases.
  - `test_odds_api_client.py`: API client behavior and formatting of Odds API responses.
- **Integration tests** (`tests/integration/`)
  - `test_end_to_end_pipeline.py`: Model training, saving, loading, API-to-ML data flow, and prediction readiness under realistic scenarios.
- **Validation tests** (`tests/validation/`)
  - `test_data_validation.py`: Data-quality and model-behavior checks (no look-ahead bias, realistic ranges, balanced target, feature importance sanity).

Run the full test suite with:

```bash
source venv/bin/activate
pytest
```

### Data & Model Assumptions

- **Historical coverage**: Engineered features are built from multi-season NBA logs up to the most recent season present in `engineered_features.csv`.
- **Players**: Original configuration focused on a curated set of high-usage players; the feature pipeline is capable of supporting a broader pool as data is added.
- **Targets**: Binary over/under outcomes relative to a player-specific threshold (`over_threshold`).
- **Markets**: Points, rebounds, and assists are fully supported; other markets can be added by extending feature engineering and training scripts.

### Current Limitations and Future Improvements

- **Player coverage**: Engineered data and thresholds are optimized for a limited set of players; extending to 30-50+ players requires updating `engineered_features.csv` and retraining models.
- **Data freshness**: Historical data stops at the last ingested season; periodic ingestion of new seasons/games and automated re-training would improve robustness.
- **Scheduling and automation**: Prediction scripts are designed for manual execution; a scheduler (cron, Airflow, etc.) and centralized logging would make this production-grade.
- **Backtesting and monitoring**: Predictions are stored in CSVs; a database-backed results log and dashboards would enable long-horizon performance monitoring and strategy iteration.
- **Deployment**: The system is optimized for local execution; containerization and simple APIs/CLI wrappers would simplify remote deployment or integration with UIs.

### Disclaimer

This repository is for **educational and research purposes only**. Sports betting involves financial risk; there is no guarantee of profit. Always gamble responsibly and comply with local laws.
