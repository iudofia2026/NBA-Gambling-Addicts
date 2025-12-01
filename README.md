# NBA Player Over/Under Predictor

A machine learning system that predicts NBA player prop bets (over/under on points, rebounds, assists) using ensemble models and live betting odds.

## What This Does

Automatically generates daily betting recommendations for NBA player props by:

1. Fetching live odds from The Odds API for today's NBA games
2. Analyzing players using trained ML models (Logistic Regression, Random Forest, XGBoost)
3. Generating predictions with confidence scores for over/under bets
4. Filtering high-confidence picks where multiple models agree
5. Providing actionable recommendations with odds and expected value

## Quick Start

### 1. Install

```bash
./setup.sh
```

The setup script creates a Python virtual environment and installs all dependencies.

### 2. Get API Key

1. Sign up at [The Odds API](https://the-odds-api.com/) - free tier: 500 requests/month
2. Copy your API key from the dashboard
3. Add to `.env` file:

```bash
echo "ODDS_API_KEY=your_key_here" > .env
```

### 3. Run Predictions

```bash
source venv/bin/activate
python src/daily_predictions.py
```

### 4. View Results

You'll see high-confidence recommendations:

```
HIGH-CONFIDENCE BETTING RECOMMENDATIONS

Stephen Curry - POINTS
   Line: 28.5
   RECOMMENDATION: OVER
   Confidence: 82%
   Model Agreement: 3/3
   Odds: -110
```

Results are also saved to `data/processed/daily_predictions_[timestamp].csv`

## Key Features

- **Ensemble ML Models**: Combines Logistic Regression, Random Forest, and XGBoost for robust predictions
- **Live Odds Integration**: Real-time betting lines from major sportsbooks via The Odds API
- **Automated Daily Pipeline**: One-command prediction generation for all games
- **Feature-Rich Analysis**: 40+ engineered features including rolling averages, rest days, opponent matchups, home/away splits
- **High-Confidence Filtering**: Only shows predictions with 60%+ confidence and model agreement
- **Comprehensive Historical Data**: Trained on NBA game logs from multiple seasons

## System Requirements

- Python 3.8+
- 4GB RAM (8GB recommended for model training)
- Internet connection for API calls
- Free API key from The Odds API

## Project Structure

```
NBA-Gambling-Addicts/
├── data/
│   ├── raw/                    # Original NBA datasets (303MB+)
│   └── processed/              # Cleaned, feature-engineered data
├── models/                     # Trained model artifacts (.pkl files)
├── notebooks/                  # Jupyter notebooks for EDA and analysis
├── src/                        # Python modules
│   ├── daily_predictions.py    # Main prediction script
│   ├── odds_api_client.py      # API integration
│   ├── ml_models.py            # Model training
│   ├── feature_engineering.py  # Feature pipeline
│   └── data_cleaning.py        # Data preprocessing
├── results/                    # Evaluation metrics and visualizations
└── setup.sh                    # Automated setup script
```

## How It Works

### Data Pipeline

1. **Data Acquisition**: Raw NBA game logs from Kaggle datasets
2. **Data Cleaning**: Handle missing values, standardize IDs, filter quality data
3. **Feature Engineering**: Generate 40+ features per game:
   - Rolling averages (3-game, 5-game, 10-game)
   - Rest days and back-to-back indicators
   - Opponent defensive metrics
   - Home/away splits
   - Season progression and usage patterns

### Model Training

- **Baseline Models**: Simple averaging approaches for comparison
- **ML Models**: Logistic Regression, Random Forest, XGBoost
- **Validation**: Time-series aware splits to prevent data leakage
- **Evaluation**: Accuracy, Brier score, precision, recall, calibration metrics

### Daily Predictions

1. Fetch today's games and player prop lines from The Odds API
2. Load trained models and historical player data
3. Generate features for each player's upcoming matchup
4. Run ensemble predictions (average of 3 model probabilities)
5. Filter for high-confidence picks (60%+ confidence, all models agree)
6. Display recommendations with confidence scores and odds

## Model Performance

Current models trained on 13 high-volume NBA players:
- **Accuracy**: 66% on test set (beats baseline by 10-15%)
- **Precision**: 68% for over predictions
- **Calibration**: Well-calibrated probabilities (Brier score < 0.25)

See `results/` directory for detailed performance metrics and visualizations.

## Tracked Players

Current model supports predictions for:
- Mikal Bridges, Buddy Hield, Harrison Barnes
- Nikola Jokic, James Harden, Rudy Gobert
- Nikola Vucevic, Tobias Harris, Devin Booker
- Karl-Anthony Towns, Jrue Holiday, Stephen Curry, Kevin Durant

## Configuration

Edit `.env` to customize:

```bash
# Required
ODDS_API_KEY=your_api_key_here

# Optional
MIN_CONFIDENCE=0.6                    # Minimum confidence for recommendations
ODDS_MARKETS=player_points,player_rebounds,player_assists
```

## Troubleshooting

### "API key required" error
```bash
# Verify .env file exists with your key
cat .env
# Should show: ODDS_API_KEY=your_key_here
```

### "No models loaded successfully"
```bash
# Train models first
python src/final_ml_models.py
```

### "No games found for today"
- Check if there are actually NBA games scheduled
- Verify API key is valid and has remaining requests
- Visit https://the-odds-api.com/account/ to check usage

## The Odds API Usage

**Free Tier Limits:**
- 500 requests per month (~16/day)
- Typical daily usage: 10-12 requests
- Monitor usage at: https://the-odds-api.com/account/

**Optimize Requests:**
- Run predictions once per day
- Best time: 2-3 hours before games start
- Check NBA schedule first to avoid wasting requests

## Academic Context

**CPSC 1710 Final Project**
*Team: Marcelo, Isiah, Gustavo*

This project demonstrates end-to-end machine learning application including:
- Feature engineering from raw sports data
- Binary classification with class imbalance handling
- Time-series validation to prevent data leakage
- Model evaluation and interpretation
- Real-world deployment with live API integration

## Limitations

- **Free API tier**: 500 requests/month (~16/day)
- **Player coverage**: Currently optimized for 13 tracked players
- **Model updates**: Retrain periodically with fresh data for best performance
- **Market types**: Points, rebounds, assists (expandable to threes, blocks, steals)

## Disclaimer

This system is for **educational purposes only**. Sports betting involves financial risk. Always gamble responsibly and do your own research before making any bets. This is not financial advice.

## License

Academic project for CPSC 1710 - Introduction to Machine Learning
Fall 2024

---

**Last Updated**: November 30, 2024
**Status**: Production-ready MVP with live deployment capabilities
**For detailed implementation status**: See [PROJECT_STATUS.md](PROJECT_STATUS.md)
