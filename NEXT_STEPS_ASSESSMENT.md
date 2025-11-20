# Next Steps Assessment: Post-Data Acquisition

**Date**: Current  
**Status**: Raw data acquired ✅ | Ready for Phase 1 completion

## Current Status

### ✅ Completed
- [x] **Data Acquisition**: All datasets downloaded and verified (24/24 files from Kaggle)
- [x] **Project Setup**: Repository structure, README, documentation
- [x] **Data Verification**: Confirmed all required files present via Kaggle comparison

### 🔄 In Progress / Next Steps

## Phase 1: Foundation (Remaining Tasks)

### 1. Data Exploration (EDA) - **IMMEDIATE PRIORITY**

**Goal**: Understand the structure, quality, and characteristics of the raw data

**Key Files to Explore**:
- `PlayerStatistics.csv` (303 MB) - Primary data source for player game logs
- `Games.csv` (9.5 MB) - Game-level context (dates, teams, scores)
- `TeamStatistics.csv` (33 MB) - Team defensive/offensive metrics
- `Players.csv` (524 KB) - Player metadata (names, IDs, positions)
- `csv/game.csv` (19 MB) - Additional game details
- `csv/play_by_play.csv` (2.1 GB) - Granular game events (may be too large for initial EDA)

**Exploration Tasks**:
1. **Data Structure Analysis**
   - Column names and data types
   - Record counts and date ranges
   - Missing value patterns
   - Data quality issues

2. **Player Statistics Analysis**
   - Distribution of points scored
   - Player activity (games played per season)
   - Identify top scorers and active players
   - Check for data consistency (player IDs, team IDs)

3. **Temporal Analysis**
   - Date range coverage
   - Season boundaries
   - Game frequency patterns
   - Identify recent vs. historical data

4. **Data Relationships**
   - Link between PlayerStatistics and Games
   - Team statistics integration
   - Player metadata connections

**Deliverable**: `notebooks/01_data_exploration.ipynb`

---

### 2. Player Selection - **HIGH PRIORITY**

**Goal**: Choose 20-50 target players for the MVP model

**Selection Criteria**:
- **High Volume**: Players with many games in dataset (more training data)
- **Consistent Playing Time**: Regular starters, not bench players
- **Point Scorers**: Focus on players who score points (not just defenders)
- **Recent Activity**: Players active in recent seasons (2023-2025)
- **Diverse Positions**: Mix of guards, forwards, centers
- **Diverse Teams**: Not all from same team
- **Predictability Range**: Mix of consistent and variable scorers

**Process**:
1. Calculate summary statistics per player:
   - Total games played
   - Average points per game
   - Standard deviation of points
   - Recent season activity (2023-2025)
   - Position information

2. Filter candidates:
   - Minimum games threshold (e.g., 50+ games)
   - Average points threshold (e.g., 10+ PPG)
   - Recent activity (games in last 2 seasons)

3. Select final 20-50 players:
   - Prioritize high-volume, active players
   - Ensure diversity in positions and teams
   - Balance between stars and role players

**Deliverable**: `data/processed/selected_players.csv` with player IDs and metadata

---

### 3. Threshold Definition - **HIGH PRIORITY**

**Goal**: Define point thresholds for each selected player

**Approach Options**:
1. **Season Average Method** (Simplest)
   - Calculate each player's season average points
   - Use average as threshold (or average ± small buffer)
   - Pros: Simple, player-specific
   - Cons: May be too easy/hard depending on player consistency

2. **Percentile-Based Method**
   - Use 50th percentile (median) as threshold
   - Or use 40th/60th percentile for more balanced classes
   - Pros: Handles skewed distributions
   - Cons: May not reflect betting lines

3. **Rolling Average Method**
   - Use recent N-game average (e.g., last 10 games)
   - Pros: Adapts to player form
   - Cons: More complex, requires temporal handling

4. **Betting Line Proxy** (Recommended for MVP)
   - Use player's season average as proxy for betting line
   - Add small variance (e.g., ±0.5 points) to create threshold
   - Pros: Mimics real-world scenario
   - Cons: Not actual betting lines

**Implementation**:
- For each selected player, calculate their overall average points
- Set threshold = average (or average + 0.5 for "over" prediction)
- Store in `data/processed/player_thresholds.csv`

**Deliverable**: `data/processed/player_thresholds.csv` with columns:
  - `personId`: Player ID
  - `threshold`: Point threshold for over/under prediction
  - `method`: How threshold was calculated

---

## Phase 2: Data Pipeline (After Phase 1)

### 4. Data Cleaning - **NEXT AFTER EDA**

**Tasks**:
- Handle missing values in PlayerStatistics
- Standardize player IDs across datasets
- Handle duplicate records
- Fix data type issues
- Validate date formats and ranges
- Handle inactive players (0 minutes, 0 points)

**Deliverable**: Cleaned datasets in `data/processed/`

---

### 5. Feature Engineering - **CORE WORK**

**Player-Level Features**:
- Rolling averages (3-game, 10-game) for:
  - Points per game
  - Minutes played
  - Field goal percentage
  - Usage rate (if available)
- Rest days calculation
- Back-to-back game indicators
- Games in last N days

**Opponent Features**:
- Opponent points allowed per game
- Opponent defensive rating (from TeamStatistics)
- Historical player performance vs. opponent

**Contextual Features**:
- Home vs. Away
- Day of week
- Month/season progression
- Recent hot/cold streak indicators

**Deliverable**: `data/processed/player_features.csv`

---

### 6. Data Integration - **CRITICAL**

**Tasks**:
- Merge PlayerStatistics with Games (for dates, home/away)
- Merge with TeamStatistics (for opponent metrics)
- Merge with Players (for metadata)
- Create final training dataset with all features + labels

**Deliverable**: `data/processed/training_data.csv`

---

### 7. Baseline Implementation - **BEFORE ML MODELS**

**Baselines to Implement**:
1. **Season Average Baseline**
   - Predict "over" if player's season average > threshold
   - Predict "under" if player's season average < threshold

2. **Rolling 5-Game Average**
   - Use last 5 games average vs. threshold

3. **Rolling 10-Game Average**
   - Use last 10 games average vs. threshold

**Purpose**: Establish performance benchmarks before building ML models

**Deliverable**: Baseline prediction functions in `src/baselines.py`

---

### 8. Train/Test Split - **BEFORE MODELING**

**Strategy**: Temporal split to prevent look-ahead bias
- Use earlier seasons for training
- Use most recent season(s) for testing
- Ensure no data leakage (no future information in training)

**Deliverable**: Split datasets in `data/processed/`

---

## Recommended Immediate Action Plan

### Week 1: Data Exploration & Player Selection
1. **Day 1-2**: Create EDA notebook
   - Load and explore PlayerStatistics.csv
   - Analyze data structure, quality, distributions
   - Identify data issues

2. **Day 3**: Player selection analysis
   - Calculate player statistics
   - Filter and select 20-50 target players
   - Document selection rationale

3. **Day 4**: Threshold definition
   - Calculate thresholds for selected players
   - Validate threshold distributions
   - Save thresholds file

### Week 2: Data Pipeline Foundation
4. **Day 5-6**: Data cleaning
   - Handle missing values
   - Standardize IDs
   - Create cleaned datasets

5. **Day 7**: Initial feature engineering
   - Implement rolling averages
   - Calculate rest days
   - Basic feature set

### Week 3+: Model Development
6. Continue with Phase 3 (Model Development) after pipeline is ready

---

## Technical Considerations

### Data Size Concerns
- `PlayerStatistics.csv` (303 MB) - Manageable in memory
- `play_by_play.csv` (2.1 GB) - May need chunked processing or sampling
- Consider using `dask` or chunked pandas for large files

### Data Quality Checks Needed
- Verify player IDs are consistent across files
- Check for duplicate game records
- Validate date formats and ranges
- Check for missing critical fields (points, gameId, personId)

### Performance Optimization
- Use efficient data types (e.g., category for strings)
- Consider parquet format for processed data (faster I/O)
- Cache intermediate results during feature engineering

---

## Success Criteria for Phase 1 Completion

- [ ] EDA notebook completed with key insights documented
- [ ] 20-50 players selected with documented rationale
- [ ] Thresholds defined and saved for all selected players
- [ ] Data quality issues identified and documented
- [ ] Ready to proceed to data cleaning and feature engineering

---

## Files to Create

### Notebooks
- `notebooks/01_data_exploration.ipynb` - Initial EDA
- `notebooks/02_player_selection.ipynb` - Player selection analysis
- `notebooks/03_threshold_definition.ipynb` - Threshold calculation

### Data Files
- `data/processed/selected_players.csv` - Selected player list
- `data/processed/player_thresholds.csv` - Point thresholds per player

### Source Code (Future)
- `src/data_loader.py` - Functions to load raw data
- `src/data_cleaner.py` - Data cleaning functions
- `src/feature_engineer.py` - Feature engineering pipeline
- `src/baselines.py` - Baseline model implementations

---

## Questions to Answer During EDA

1. **Data Coverage**: What date range do we have? How many seasons?
2. **Player Coverage**: How many unique players? How many games per player?
3. **Data Completeness**: What percentage of games have complete player stats?
4. **Point Distributions**: What's the distribution of points scored? Any outliers?
5. **Temporal Patterns**: Are there trends over time? Seasonality?
6. **Missing Data**: Which fields have missing values? How should we handle them?
7. **Data Relationships**: How do the different datasets connect? Any join issues?

---

## Next Immediate Steps

1. **Create EDA notebook** (`notebooks/01_data_exploration.ipynb`)
2. **Load and explore PlayerStatistics.csv** - Start with basic statistics
3. **Analyze Games.csv** - Understand game context
4. **Identify data quality issues** - Document problems
5. **Begin player selection process** - Calculate player statistics

**Start with**: Creating the EDA notebook and loading the first dataset!
