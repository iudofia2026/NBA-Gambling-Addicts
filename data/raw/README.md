# Raw Data Directory

This directory contains the raw NBA datasets downloaded from Kaggle.

## Required Datasets

1. **Historical NBA Data and Player Box Scores**
   - Source: https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores/data
   - Key files: `games_details.csv`, team statistics, player game logs

2. **Basketball Reference Dataset**
   - Source: https://www.kaggle.com/datasets/wyattowalsh/basketball/data
   - Key files: Team statistics, advanced metrics

## Setup Instructions

1. Install Kaggle CLI: `pip install kaggle`
2. Set up Kaggle credentials (kaggle.json in ~/.kaggle/)
3. Download datasets:
   ```bash
   kaggle datasets download -d eoinamoore/historical-nba-data-and-player-box-scores
   kaggle datasets download -d wyattowalsh/basketball
   ```
4. Extract files to this directory

## File Structure (After Download)
```
raw/
├── games_details.csv          # Individual player game statistics
├── games.csv                  # Game-level information
├── players.csv                # Player information
├── teams.csv                  # Team information
├── TeamStatistics.csv         # Team defensive/offensive stats
└── other_basketball_files/    # Additional basketball reference data
```