# Raw Data Directory

This directory contains the unmodified NBA datasets pulled from Kaggle. The large files
are ignored by git to keep the repository lightweight.

## Required Datasets

1. **Historical NBA Data and Player Box Scores**
   - Source: https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores/data
   - Key files: `games_details.csv`, `games.csv`, `players.csv`, `teams.csv`

2. **Basketball Reference Dataset**
   - Source: https://www.kaggle.com/datasets/wyattowalsh/basketball/data
   - Key files: Team statistics, advanced metrics, play-by-play data

## Manual Download Instructions

1. Install the Kaggle CLI: `pip install kaggle`
2. Configure Kaggle credentials (kaggle.json in `~/.kaggle/` or export `KAGGLE_USERNAME`/`KAGGLE_KEY`)
3. Download each dataset:
   ```bash
   kaggle datasets download -d eoinamoore/historical-nba-data-and-player-box-scores -p data/raw
   kaggle datasets download -d wyattowalsh/basketball -p data/raw
   ```
4. Extract the resulting `.zip` files directly into this directory

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