## Implementation Overview

- **Goal**: Predict NBA player prop outcomes (over/under) using historical data, engineered features, and ensemble ML models, integrated with live betting odds.
- **Components**: Data preparation, feature engineering, model training, prediction systems, Odds API client, and test/validation layers.
- **Execution modes**: Standard daily predictions, final context-rich system, and an experimental complete-iterations predictor.

## Data Pipeline

- **Input sources**
  - Processed historical dataset in `data/processed/engineered_features.csv` produced from raw NBA box score and schedule data.
  - Auxiliary processed artifacts (cleaning summaries, age analysis, feature importance) used for analysis and reporting.
- **Cleaning and preprocessing** (`data_cleaning.py`)
  - Standardizes date formats and identifiers.
  - Handles missing values and inconsistent records.
  - Ensures per-player chronological ordering of games.
- **Feature engineering** (`feature_engineering.py`, `advanced_features_v1.py`, `evidence_features_v4.py`, `matchup_analytics_v5.py`, `advanced_analytics_v6.py`)
  - Rolling averages over multiple windows (3/5/10 games) for points, minutes, assists, and shooting metrics.
  - Rest and schedule indicators (days of rest, back-to-back flags, games in recent windows).
  - Trend and volatility metrics (over-streaks, threshold deltas, minutes trends, hot/cold streaks).
  - Contextual attributes (season progression, month, weekday/weekend, home vs away, game type/label).
  - Evidence-based factors (home-court advantage, recent context, opponent strength) and matchup analytics (player vs team history, role, usage, matchup composite scores).
  - Advanced seasonal, peak-performance, team-dynamics, and situational-pressure features (streak detection, clutch/big-margin performance, team momentum, usage share, seasonal improvement).

## Modeling and Training

- **Leakage-safe training** (`final_ml_models.py`)
  - Loads `engineered_features.csv` and sorts by player and date.
  - Removes direct target and outcome columns (e.g., `points`, `points_vs_threshold`, `points_per_minute`, `over_threshold`, `gameDate`, name/ID fields, current-game box-score stats) to prevent leakage.
  - Retains only features that are knowable before tip-off (rolling windows, schedule features, trends, contextual and categorical features).
  - Constructs temporal train/test splits based on game date (80/20 by time).
- **Model types**
  - Logistic Regression with standardized inputs and class-weight handling.
  - Random Forest with tuned depth, split criteria, and class weights, using parallel training.
  - Optional XGBoost model support via `ml_models.py` when the dependency is available.
- **Feature preparation** (`ml_models.py`)
  - Temporal splits per `create_temporal_train_test_split` to avoid look-ahead bias.
  - `prepare_features` encodes categorical columns with `LabelEncoder` instances shared between train and test.
  - Fills missing values and coerces all feature columns to numeric types.
  - Returns feature matrices, targets, feature column lists, and encoders for downstream persistence.
- **Evaluation** (`ml_models.py`, `final_ml_models.py`)
  - Computes accuracy, precision, recall, F1, ROC AUC, and log loss on the hold-out period.
  - Extracts feature importances (Random Forest) and saves aggregate model comparison tables to `data/processed` (e.g., `clean_ml_results.csv`).
  - Validates that models outperform simple baselines (e.g., 5-game rolling average) and that metrics remain in valid ranges.
- **Persistence** (`models/`)
  - Saves trained models (e.g., `logistic_regression_model.pkl`, `random_forest_model.pkl`, `xgboost_model.pkl`).
  - Stores `feature_columns.pkl` and `label_encoders.pkl` to align training and inference feature spaces.

## Prediction Systems

- **`DailyPredictor` (`daily_predictions.py`)**
  - Loads serialized models and shared feature metadata from `models/`.
  - Loads historical feature data from `data/processed/engineered_features.csv`.
  - For each player prop:
    - Locates recent player history and constructs a game-context feature vector aligned with `feature_columns.pkl`.
    - Applies stored `LabelEncoder` instances to categorical features; fills missing values.
    - Obtains per-model predictions and probabilities, and aggregates them into an ensemble (majority vote plus average probabilities).
  - Computes:
    - Binary recommendation (OVER/UNDER).
    - Confidence score (probability associated with the recommended side).
    - Model agreement ratio and per-model prediction breakdown.
  - Filters to high-confidence picks by requiring confidence > 0.6 and full agreement (`over_percentage` in {0.0, 1.0}).
  - Displays structured console output and writes complete prediction tables to `data/processed/daily_predictions_<timestamp>.csv`.

- **`FinalNBAPredictionsSystem` (`final_predictions_system.py`)**
  - Uses historical features to derive baseline performance and additional contextual layers:
    - Player form (streaks, trends, variability-based confidence).
    - Team chemistry (team momentum, player usage share, chemistry impact).
    - Individual matchup history (average points vs opponent, over-rate, efficiency, trend, and sample-size confidence).
  - Combines form, chemistry, matchup, and baseline into a weighted predicted value and confidence score.
  - Generates recommendations for points and mapped rebounds markets, attaches interpretable insights (predicted value vs line, streak description, chemistry summary, matchup stats).
  - Outputs final recommendations with richer narrative context and writes `final_predictions_<timestamp>.csv`.

- **`CompleteNBAPredictor` and advanced analytics (`advanced_analytics_v6.py`)**
  - Integrates evidence-based features, matchup analytics, and advanced seasonal/pressure features into a single predictor.
  - Computes a total adjustment to a baseline based on:
    - Evidence-based adjustment (e.g., home/away, rest, recent form).
    - Composite matchup scores (player vs opponent team characteristics).
    - Advanced analytics composite adjustment (seasonal improvement, peak usage, team momentum, clutch performance).
  - Produces an ultimate predicted value and combined confidence, filtering out low-confidence cases.
  - Outputs enriched CSVs (`ultimate_predictions_<timestamp>.csv`) for experimentation and analysis.

## Odds API Integration

- **Client** (`odds_api_client.py`)
  - Wraps The Odds API for NBA game and player prop retrieval.
  - Implements request construction, error handling, and basic rate-limit awareness (e.g., using `x-requests-remaining` header).
  - Exposes methods such as `get_all_todays_player_props` and `get_player_props_for_game`.
  - Formats raw API responses into ML-ready tables via `format_for_ml_pipeline` by:
    - Normalizing field names (e.g., `fullName`, `gameDate`, `prop_line`, `market_type`, `home_team`, `away_team`, odds and bookmaker fields).
    - Casting numeric and datetime fields to appropriate dtypes.

## Testing and Validation

- **Unit tests**
  - `tests/unit/test_ml_models.py`: Verifies data loading behavior, temporal splitting, feature preparation (including leakage exclusion and categorical encoding), model evaluation utilities, trainer behavior, and the `ScaledLogisticRegression` wrapper.
  - `tests/unit/test_daily_predictions.py`: Covers initialization paths, model/feature/encoder loading, historical data access, feature generation for predictions, ensemble logic and disagreement handling, prediction display, and edge cases (NaNs, missing features, unknown players).
  - `tests/unit/test_odds_api_client.py`: Validates API client behavior, request/response handling, and ML-format conversion (column presence and type sanity checks).

- **Integration tests** (`tests/integration/test_end_to_end_pipeline.py`)
  - Emulate full training workflows: mock data ingestion, temporal splitting, feature prep, model training, and artifact persistence.
  - Emulate full prediction workflows: model loading, historical data access, mocked API calls, data formatting, prediction generation, and CSV persistence.
  - Validate that training and prediction stages can be chained realistically with mocked external dependencies.

- **Validation tests** (`tests/validation/test_data_validation.py`)
  - Check data ranges (points, minutes, prop lines, odds) for realism.
  - Confirm absence of look-ahead bias in temporal splits and rolling features.
  - Enforce reasonable class balance for the target and sufficient variability in predictions.
  - Verify feature importance distributions, metric ranges, and basic model performance (better than random, non-degenerate predictions).

## Operational Characteristics

- **Runtime behavior**
  - Daily prediction scripts complete within seconds to minutes on typical hardware for a days worth of player props.
  - Requests to The Odds API remain within free-tier limits when run once per day.
- **Execution patterns**
  - Manual CLI execution of training and prediction scripts via Python entry points in `src/`.
  - Model artifacts and prediction outputs persisted as files for later inspection and analysis.

## Known Limitations

- **Player and data coverage**
  - Engineered features and thresholds currently center on a limited set of high-usage players.
  - Historical data is capped at the most recent ingested season and does not automatically update as new games are played.
- **Operational constraints**
  - No built-in job scheduler or orchestration for automatic daily runs.
  - No centralized storage of predictions and outcomes beyond CSV exports.
  - No live performance monitoring or alerting when models degrade.
- **Data and feature gaps**
  - No direct integration of injury or lineup information; dependence on the API to filter out inactive players.
  - Limited opponent-specific contextual features beyond those engineered from historical box scores.
  - No access to historical odds for full backtesting against market lines.

## Improvement Opportunities

- **Data and coverage**
  - Extend `engineered_features.csv` to additional players and seasons, including automatic ingestion of new games.
  - Introduce data-versioning and external storage for large raw datasets.
- **Modeling and evaluation**
  - Add automated retraining pipelines (e.g., scheduled weekly/monthly runs) with tracked model versions.
  - Incorporate calibration layers (Platt scaling, isotonic regression) and more granular evaluation by context (player archetype, game type, schedule conditions).
  - Integrate richer opponent/team defensive profiles and on/off splits into the feature space.
- **System and operations**
  - Wrap prediction flows in scheduled jobs (cron, workflow managers) with logging and alerting.
  - Persist predictions and realized outcomes to a database for long-term tracking and backtesting.
  - Add containerization and lightweight APIs/CLIs to expose predictions to external services or user interfaces.
- **Product and UX**
  - Build a simple web or notebook dashboard for exploring daily recommendations and historical performance.
  - Surface model confidence, feature attributions, and scenario analysis tools for power users.
