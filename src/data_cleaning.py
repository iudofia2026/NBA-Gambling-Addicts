"""
NBA Data Cleaning Script

This module cleans and preprocesses the raw NBA data for our over/under prediction models.
Includes age-based adjustments and filters for our 13 target players.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Our target players with corrected dataset names
TARGET_PLAYERS = {
    'Mikal Bridges': 'Mikal Bridges',
    'Buddy Hield': 'Buddy Hield',
    'Harrison Barnes': 'Harrison Barnes',
    'Nikola Jokić': 'Nikola Jokic',
    'James Harden': 'James Harden',
    'Rudy Gobert': 'Rudy Gobert',
    'Nikola Vučević': 'Nikola Vucevic',
    'Tobias Harris': 'Tobias Harris',
    'Devin Booker': 'Devin Booker',
    'Karl-Anthony Towns': 'Karl-Anthony Towns',
    'Jrue Holiday': 'Jrue Holiday',
    'Stephen Curry': 'Stephen Curry',
    'Kevin Durant': 'Kevin Durant'
}

def load_raw_data():
    """Load all raw datasets needed for cleaning."""

    print("=== NBA DATA CLEANING PIPELINE ===")
    print("Loading raw datasets...")

    # Load main datasets
    player_stats = pd.read_csv('../data/raw/PlayerStatistics.csv', low_memory=False)
    games = pd.read_csv('../data/raw/Games.csv')
    players_info = pd.read_csv('../data/raw/Players.csv')
    team_stats = pd.read_csv('../data/raw/TeamStatistics.csv')

    # Load our preprocessed files
    thresholds = pd.read_csv('../data/processed/player_thresholds.csv')
    age_curves = pd.read_csv('../data/processed/age_performance_curves.csv')

    print(f"✓ Player statistics: {len(player_stats):,} records")
    print(f"✓ Games: {len(games):,} records")
    print(f"✓ Player info: {len(players_info):,} players")
    print(f"✓ Team stats: {len(team_stats):,} records")
    print(f"✓ Player thresholds: {len(thresholds)} players")
    print(f"✓ Age performance curves: {len(age_curves)} age points")

    return {
        'player_stats': player_stats,
        'games': games,
        'players_info': players_info,
        'team_stats': team_stats,
        'thresholds': thresholds,
        'age_curves': age_curves
    }

def clean_player_statistics(player_stats, players_info):
    """Clean and filter the main player statistics data."""

    print("\nCleaning player statistics...")

    # Create full name column
    player_stats['fullName'] = player_stats['firstName'].astype(str) + ' ' + player_stats['lastName'].astype(str)
    players_info['fullName'] = players_info['firstName'].astype(str) + ' ' + players_info['lastName'].astype(str)

    original_count = len(player_stats)

    # Basic data quality filters
    print("Applying data quality filters...")

    # 1. Remove games where player didn't play (DNP)
    played_games = player_stats[player_stats['numMinutes'] > 0].copy()
    print(f"  - Removed DNP games: {original_count - len(played_games):,} games")

    # 2. Remove games with missing essential data
    essential_cols = ['points', 'numMinutes', 'gameDate']
    complete_games = played_games.dropna(subset=essential_cols).copy()
    print(f"  - Removed incomplete records: {len(played_games) - len(complete_games):,} games")

    # 3. Remove obvious outliers (e.g., negative values, impossible values)
    valid_games = complete_games[
        (complete_games['points'] >= 0) &
        (complete_games['points'] <= 100) &  # Max possible in a game
        (complete_games['numMinutes'] >= 0) &
        (complete_games['numMinutes'] <= 60)  # Max possible in a game
    ].copy()
    print(f"  - Removed outliers: {len(complete_games) - len(valid_games):,} games")

    # 4. Filter to our target players only
    target_games = valid_games[
        valid_games['fullName'].isin(list(TARGET_PLAYERS.values()))
    ].copy()
    print(f"  - Filtered to target players: {len(target_games):,} games from {target_games['fullName'].nunique()} players")

    # 5. Add age information
    print("Adding age information...")

    # Extract years for age calculation
    def extract_year(date_str):
        try:
            return int(str(date_str)[:4]) if not pd.isna(date_str) else None
        except:
            return None

    target_games['game_year'] = target_games['gameDate'].apply(extract_year)
    players_info['birth_year'] = players_info['birthdate'].apply(extract_year)

    # Merge with birth year
    target_games = target_games.merge(
        players_info[['fullName', 'birth_year']],
        on='fullName',
        how='left'
    )

    # Calculate age
    target_games['age_at_game'] = target_games['game_year'] - target_games['birth_year']

    # Filter to reasonable age range
    aged_games = target_games[
        (target_games['age_at_game'] >= 18) |
        (target_games['age_at_game'].isna())  # Keep games without age data
    ].copy()

    ages_available = aged_games['age_at_game'].notna().sum()
    print(f"  - Age data available for {ages_available:,} games")

    # 6. Convert and clean data types
    print("Standardizing data types...")

    # Ensure numeric columns are numeric
    numeric_cols = [
        'points', 'numMinutes', 'assists', 'reboundsTotal', 'fieldGoalsMade',
        'fieldGoalsAttempted', 'threePointersMade', 'threePointersAttempted',
        'freeThrowsMade', 'freeThrowsAttempted', 'steals', 'blocks', 'turnovers'
    ]

    for col in numeric_cols:
        if col in aged_games.columns:
            aged_games[col] = pd.to_numeric(aged_games[col], errors='coerce')

    # Ensure boolean columns are boolean
    bool_cols = ['home', 'win']
    for col in bool_cols:
        if col in aged_games.columns:
            aged_games[col] = aged_games[col].astype('boolean')

    # Convert game date to datetime
    aged_games['gameDate'] = pd.to_datetime(aged_games['gameDate'], errors='coerce')

    print(f"✓ Final cleaned dataset: {len(aged_games):,} games")
    print(f"  - Players: {aged_games['fullName'].nunique()}")
    print(f"  - Date range: {aged_games['gameDate'].min()} to {aged_games['gameDate'].max()}")

    return aged_games

def add_target_labels(cleaned_data, thresholds):
    """Add over/under target labels based on our defined thresholds."""

    print("\nAdding over/under target labels...")

    # Create threshold lookup
    threshold_lookup = thresholds.set_index('player_dataset_name')['proposed_threshold'].to_dict()

    # Add threshold for each player
    cleaned_data['player_threshold'] = cleaned_data['fullName'].map(threshold_lookup)

    # Create binary over/under labels
    cleaned_data['over_threshold'] = (cleaned_data['points'] > cleaned_data['player_threshold']).astype(int)

    # Verify label distribution
    label_dist = cleaned_data.groupby('fullName').agg({
        'over_threshold': ['count', 'mean'],
        'player_threshold': 'first'
    }).round(3)

    label_dist.columns = ['total_games', 'over_rate', 'threshold']

    print("Over/Under label distribution by player:")
    print(label_dist)

    # Overall distribution
    overall_over_rate = cleaned_data['over_threshold'].mean()
    print(f"\nOverall over rate: {overall_over_rate:.3f} ({overall_over_rate*100:.1f}%)")

    return cleaned_data

def add_age_based_features(cleaned_data, age_curves):
    """Add age-based performance features."""

    print("\nAdding age-based features...")

    # Create age performance lookup
    age_lookup = age_curves.set_index('age_at_game').to_dict()

    # Add expected performance based on age
    cleaned_data['age_expected_points'] = cleaned_data['age_at_game'].map(age_lookup['avg_points'])
    cleaned_data['age_expected_minutes'] = cleaned_data['age_at_game'].map(age_lookup['avg_minutes'])

    # Create relative performance features
    cleaned_data['points_vs_age_expected'] = cleaned_data['points'] - cleaned_data['age_expected_points']
    cleaned_data['minutes_vs_age_expected'] = cleaned_data['numMinutes'] - cleaned_data['age_expected_minutes']

    # Age categories
    cleaned_data['age_category'] = pd.cut(
        cleaned_data['age_at_game'],
        bins=[0, 23, 27, 30, 50],
        labels=['Young', 'Developing', 'Prime', 'Veteran'],
        include_lowest=True
    )

    # Career stage (approximate years in league)
    cleaned_data['years_in_league_approx'] = cleaned_data['age_at_game'] - 19
    cleaned_data['career_stage'] = pd.cut(
        cleaned_data['years_in_league_approx'],
        bins=[0, 4, 8, 12, 30],
        labels=['Early', 'Developing', 'Prime', 'Veteran'],
        include_lowest=True
    )

    # Count of age-based features added
    age_features_count = cleaned_data['age_expected_points'].notna().sum()
    print(f"✓ Age-based features added to {age_features_count:,} games")

    return cleaned_data

def add_basic_features(cleaned_data):
    """Add basic engineered features before advanced feature engineering."""

    print("\nAdding basic derived features...")

    # Shooting efficiency features
    cleaned_data['fg_percentage'] = np.where(
        cleaned_data['fieldGoalsAttempted'] > 0,
        cleaned_data['fieldGoalsMade'] / cleaned_data['fieldGoalsAttempted'],
        0
    )

    cleaned_data['three_point_percentage'] = np.where(
        cleaned_data['threePointersAttempted'] > 0,
        cleaned_data['threePointersMade'] / cleaned_data['threePointersAttempted'],
        0
    )

    cleaned_data['free_throw_percentage'] = np.where(
        cleaned_data['freeThrowsAttempted'] > 0,
        cleaned_data['freeThrowsMade'] / cleaned_data['freeThrowsAttempted'],
        0
    )

    # Usage indicators
    cleaned_data['points_per_minute'] = np.where(
        cleaned_data['numMinutes'] > 0,
        cleaned_data['points'] / cleaned_data['numMinutes'],
        0
    )

    # Total shots
    cleaned_data['total_shot_attempts'] = (
        cleaned_data['fieldGoalsAttempted'].fillna(0) +
        cleaned_data['freeThrowsAttempted'].fillna(0)
    )

    # Extract temporal features
    cleaned_data['month'] = cleaned_data['gameDate'].dt.month
    cleaned_data['day_of_week'] = cleaned_data['gameDate'].dt.dayofweek
    cleaned_data['is_weekend'] = cleaned_data['day_of_week'].isin([5, 6]).astype(int)

    print("✓ Added basic derived features")

    return cleaned_data

def save_cleaned_data(cleaned_data):
    """Save the cleaned data to processed folder."""

    print("\nSaving cleaned data...")

    # Ensure directory exists
    os.makedirs('../data/processed', exist_ok=True)

    # Save main cleaned dataset
    output_file = '../data/processed/cleaned_player_data.csv'
    cleaned_data.to_csv(output_file, index=False)

    print(f"✓ Saved cleaned data: {output_file}")
    print(f"  - Shape: {cleaned_data.shape}")
    print(f"  - Memory usage: {cleaned_data.memory_usage(deep=True).sum() / (1024**2):.1f} MB")

    # Create summary statistics
    summary_stats = {
        'total_games': len(cleaned_data),
        'unique_players': cleaned_data['fullName'].nunique(),
        'date_range_start': str(cleaned_data['gameDate'].min()),
        'date_range_end': str(cleaned_data['gameDate'].max()),
        'over_rate': cleaned_data['over_threshold'].mean(),
        'avg_points': cleaned_data['points'].mean(),
        'features_count': len(cleaned_data.columns),
        'cleaning_timestamp': datetime.now().isoformat()
    }

    # Save summary
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv('../data/processed/cleaning_summary.csv', index=False)

    print("✓ Saved cleaning summary")

    return summary_stats

def main():
    """Main data cleaning workflow."""

    # Step 1: Load raw data
    raw_data = load_raw_data()

    # Step 2: Clean player statistics
    cleaned_stats = clean_player_statistics(
        raw_data['player_stats'],
        raw_data['players_info']
    )

    # Step 3: Add target labels (over/under)
    labeled_data = add_target_labels(cleaned_stats, raw_data['thresholds'])

    # Step 4: Add age-based features
    aged_data = add_age_based_features(labeled_data, raw_data['age_curves'])

    # Step 5: Add basic derived features
    featured_data = add_basic_features(aged_data)

    # Step 6: Save cleaned data
    summary = save_cleaned_data(featured_data)

    print("\n=== DATA CLEANING COMPLETE ===")
    print(f"✅ Processed {summary['total_games']:,} games for {summary['unique_players']} players")
    print(f"✅ Over rate: {summary['over_rate']:.1%} (well balanced)")
    print(f"✅ Features: {summary['features_count']} columns available for modeling")
    print(f"✅ Ready for feature engineering and baseline models")

    return featured_data

if __name__ == "__main__":
    main()