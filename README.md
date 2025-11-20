# NBA Player Over/Under Predictor

**CPSC 1710 Final Project**
*Team: Marcelo, Isiah, Gustavo*

A machine learning system that predicts whether NBA players will score "over" or "under" specific point thresholds in their next game. This project addresses a classic problem in fantasy sports and betting by creating a binary classifier trained on historical NBA data.

## Project Overview

### What We're Building
- **Goal**: Binary classifier to predict if NBA players will score above or below betting line thresholds (e.g., 25.5 points)
- **Purpose**: Test against real-world betting lines during the NBA season for immediate feedback validation
- **Scope**: Focus on ~20-50 key NBA players for manageable but meaningful predictions

### Why This Matters
- **Fast Feedback Loop**: Daily NBA games provide constant validation opportunities
- **Real-World Relevance**: Directly comparable to actual sports betting and fantasy performance
- **ML Application**: Perfect use case for classification with complex, hidden variables (fatigue, matchups, hot streaks)
- **Academic Value**: Demonstrates feature engineering, model selection, and evaluation under distribution shift

### How We're Approaching It
- **Problem Type**: Binary classification ("over" = 1, "under" = 0)
- **Models**: Start with Logistic Regression → Random Forest/XGBoost
- **Evaluation**: Nightly backtesting with accuracy, Brier score, and calibration metrics
- **Baseline Comparisons**: Season averages vs. thresholds, rolling N-game averages

## Technical Architecture

### Data Sources
1. **Historical NBA Data & Player Box Scores** (Kaggle)
   - `PlayerStatistics.csv` - Individual player game logs with points, minutes, usage
   - `Games.csv` - Game-level information (home/away, dates)
   - `TeamStatistics.csv` - Team offensive/defensive metrics

2. **Basketball Reference Dataset** (Kaggle)
   - `game.csv` - Detailed game information
   - `play_by_play.csv` - Granular game events
   - Additional team and player metrics

### Feature Engineering Pipeline

#### Player-Level Features
- **Rolling Averages**: 3-game and 10-game moving averages for:
  - Points per game
  - Minutes played
  - Usage rate
  - Field goal percentage
- **Rest & Schedule**:
  - Days of rest since last game
  - Back-to-back game indicators
  - Games in last N days (fatigue proxy)

#### Opponent Features
- **Defensive Strength**:
  - Opponent points allowed per game
  - Opponent defensive rating
  - Historical performance vs. similar players
- **Pace Factors**:
  - Team pace (possessions per game)
  - Recent pace trends

#### Contextual Features
- **Game Context**:
  - Home vs. Away
  - Day of week
  - Month/season progression
- **Player Context**:
  - Season averages relative to career norms
  - Recent hot/cold streak indicators

### Model Development Strategy

#### Phase 1: Baseline Models
```python
# Simple baselines for comparison
1. Season average vs. threshold
2. Last 5 games average vs. threshold
3. Last 10 games average vs. threshold
```

#### Phase 2: ML Models
```python
# Progressive model complexity
1. Logistic Regression (interpretable, fast)
2. Random Forest (handles interactions, robust)
3. XGBoost (gradient boosting, high performance)
```

#### Phase 3: Model Optimization
- **Cross-Validation**: Player-aware splits to prevent data leakage
- **Hyperparameter Tuning**: Grid/random search with validation
- **Calibration**: Isotonic regression for probability calibration
- **Feature Selection**: Recursive feature elimination, importance analysis

### Evaluation Framework

#### Metrics
- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Probability**: Brier Score (primary), Log-Loss
- **Calibration**: Reliability plots, calibration error
- **Practical**: ROI simulation against betting lines

#### Validation Strategy
```python
# Time-aware validation to prevent look-ahead bias
1. Walk-forward validation (temporal splits)
2. Player-stratified CV (prevent player leakage)
3. Hold-out test set (final evaluation)
4. Live forward testing (during season)
```

#### Error Analysis
- **Performance by Player**: Identify which players are most/least predictable
- **Performance by Context**: Home vs away, rest days, opponent strength
- **Feature Importance**: SHAP values for model interpretability
- **Failure Case Analysis**: When and why predictions fail

## Project Timeline

### Phase 1: Foundation (Nov 10-14) - 6-8 hours
- [x] **Data Acquisition**: Download and extract NBA datasets
- [x] **Project Setup**: Repository structure, README, initial documentation
- [ ] **Data Exploration**: Initial EDA on key files (PlayerStatistics.csv, Games.csv)
- [ ] **Player Selection**: Choose 20-50 target players for MVP
- [ ] **Threshold Definition**: Define point thresholds for each player

### Phase 2: Data Pipeline (Nov 15-21) - 10-12 hours
- [ ] **Data Cleaning**: Handle missing values, data quality issues
- [ ] **Feature Engineering**: Implement rolling averages, rest days, opponent metrics
- [ ] **Data Integration**: Merge player stats with game context and opponent data
- [ ] **Baseline Implementation**: Season averages and rolling window baselines
- [ ] **Train/Test Split**: Implement temporal splits for valid evaluation

### Phase 3: Model Development (Nov 22-28) - 8-10 hours
- [ ] **Initial Models**: Logistic regression with basic features
- [ ] **Model Pipeline**: Cross-validation framework with player-aware splits
- [ ] **Advanced Models**: Random Forest and XGBoost implementation
- [ ] **Evaluation Framework**: Metrics calculation and comparison system
- [ ] **Initial Dashboard**: Simple visualization of predictions vs actuals

### Phase 4: Optimization & Analysis (Nov 29-Dec 5) - 12-14 hours
- [ ] **Hyperparameter Tuning**: Grid search for optimal model parameters
- [ ] **Feature Selection**: Identify most predictive features
- [ ] **Probability Calibration**: Improve prediction confidence reliability
- [ ] **Error Analysis**: Deep dive into model failures and edge cases
- [ ] **Performance Analysis**: Player-specific and context-specific performance

### Phase 5: Documentation & Presentation (Dec 6-12) - 10-12 hours
- [ ] **Dashboard Polish**: Clean, interpretable visualization of results
- [ ] **Documentation**: Comprehensive technical documentation
- [ ] **Report Writing**: Academic report with methodology and findings
- [ ] **Presentation Prep**: Slides and live demo preparation
- [ ] **Code Organization**: Clean, documented, reproducible code

### Phase 6: Final Delivery (Finals Week) - 8-10 hours
- [ ] **Presentation Rehearsal**: Practice and refine presentation
- [ ] **Final Report**: Complete academic report submission
- [ ] **Code Submission**: Final code package with documentation

## Team Responsibilities

### Marcelo
- **Primary**: Data wrangling, exploration, and evaluation framework
- **Specific Tasks**:
  - Raw data processing and cleaning
  - Exploratory data analysis and insights
  - Baseline model implementation
  - Nightly evaluation logging system
  - Initial model training and validation

### Isiah
- **Primary**: Feature engineering, model tuning, and visualization
- **Specific Tasks**:
  - Rolling average and contextual feature creation
  - Model hyperparameter optimization
  - Probability calibration implementation
  - Dashboard and visualization development
  - Presentation assets and slides

### Gustavo
- **Primary**: Model evaluation, debugging, and error analysis
- **Specific Tasks**:
  - Cross-validation framework implementation
  - Model performance debugging
  - Error analysis and failure case identification
  - Feature importance and model interpretability
  - Documentation and code organization

### Shared Responsibilities
- **All Team Members**:
  - Weekly progress meetings and coordination
  - Risk mitigation and problem-solving
  - Final report writing and editing
  - Presentation delivery and Q&A

## Risk Management

### Technical Risks
1. **Data Quality Issues**
   - *Risk*: Missing games, inconsistent player IDs, data gaps
   - *Mitigation*: Robust data validation, multiple data sources, graceful handling of missing data

2. **Class Imbalance**
   - *Risk*: Skewed over/under distributions for certain thresholds
   - *Mitigation*: Class weighting, threshold adjustment, stratified sampling

3. **Overfitting to Star Players**
   - *Risk*: Model performs well on a few players but poorly on others
   - *Mitigation*: Player-stratified validation, diverse player selection, regularization

4. **Distribution Shift**
   - *Risk*: Player roles, team strategies change during season
   - *Mitigation*: Recency weighting, adaptive thresholds, continuous monitoring

### Project Risks
1. **Timeline Delays**
   - *Risk*: Data processing takes longer than expected
   - *Mitigation*: Front-load data work, have backup simplified approaches

2. **Threshold Availability**
   - *Risk*: Difficulty obtaining real betting lines
   - *Mitigation*: Use player-specific season averages as proxy thresholds

3. **Model Complexity**
   - *Risk*: Over-engineering leads to poor performance or interpretability
   - *Mitigation*: Start simple (logistic regression), add complexity incrementally

## Success Metrics

### MVP Success Criteria
- [ ] **Functional Classifier**: Working binary classifier for 20+ players
- [ ] **Baseline Comparison**: Outperform simple averaging baselines
- [ ] **Evaluation Framework**: Daily prediction and evaluation for 1+ weeks
- [ ] **Reproducible Results**: Clear documentation and runnable code
- [ ] **Academic Deliverables**: Complete report and presentation ready

### Stretch Goals
- [ ] **Strong Performance**: >55% accuracy with good calibration
- [ ] **Real-time Predictions**: System for generating daily predictions
- [ ] **Comprehensive Analysis**: Deep insights into predictability patterns
- [ ] **Feature Insights**: Clear understanding of what drives performance
- [ ] **Practical Application**: Demonstrate potential for real-world use

## Getting Started

### Prerequisites
```bash
# Required packages
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

### Quick Start
```bash
# 1. Clone repository
git clone <repo-url>
cd NBA-Gambling-Addicts

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the raw datasets from Kaggle (see data/raw/README.md for details)
#    Example:
#    kaggle datasets download -d eoinamoore/historical-nba-data-and-player-box-scores -p data/raw
#    kaggle datasets download -d wyattowalsh/basketball -p data/raw
#    unzip *.zip

# 4. Start with exploratory notebook
jupyter notebook notebooks/01_data_exploration.ipynb

# 5. Follow the phase-by-phase development plan
```

### Repository Structure
```
NBA-Gambling-Addicts/
├── data/
│   ├── raw/                    # Original datasets from Kaggle
│   └── processed/              # Cleaned, feature-engineered data
├── notebooks/                  # Jupyter notebooks for analysis
├── src/                       # Python modules and scripts
├── models/                    # Trained model artifacts
├── results/                   # Evaluation results and figures
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

## Academic Integrity

This project is developed for CPSC 1710 coursework. All data sources are publicly available, and methodologies follow standard machine learning practices. The team commits to:

- Original implementation and analysis
- Proper citation of data sources and methodological references
- Transparent reporting of results, including negative findings
- Collaborative development with clear individual contributions
- Adherence to course guidelines and academic standards

---

**Contact**: [Team member emails/GitHub profiles]
**Course**: CPSC 1710 - Introduction to Machine Learning
**Institution**: [University Name]
**Semester**: Fall 2024
