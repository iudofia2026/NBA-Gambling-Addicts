# Processed Data Directory

This directory contains cleaned and engineered datasets ready for model training.

## Generated Files

- `player_features.csv` - Engineered features for each player-game
- `training_data.csv` - Final dataset with labels for model training
- `player_thresholds.csv` - Player-specific point thresholds for predictions
- `team_defensive_stats.csv` - Processed opponent defensive metrics

## Feature Engineering Pipeline

Features include:
- Rolling averages (3-game, 10-game) for points, minutes, usage
- Days of rest and back-to-back game indicators
- Opponent defensive strength metrics
- Home/away game indicators
- Recent pace and usage rate trends