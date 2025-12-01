"""
NBA Age-Based Performance Analysis

This module analyzes how player performance changes with age across all NBA players
to create robust age-based features for our prediction models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_merge_data():
    """Load player statistics and merge with birth date information."""

    print("=== NBA AGE-BASED PERFORMANCE ANALYSIS ===")
    print("Loading and merging datasets...")

    # Load datasets
    player_stats = pd.read_csv('../data/raw/PlayerStatistics.csv', low_memory=False)
    players_info = pd.read_csv('../data/raw/Players.csv')

    # Create full name mapping
    player_stats['fullName'] = player_stats['firstName'].astype(str) + ' ' + player_stats['lastName'].astype(str)
    players_info['fullName'] = players_info['firstName'].astype(str) + ' ' + players_info['lastName'].astype(str)

    print(f"✓ Player statistics: {len(player_stats):,} records")
    print(f"✓ Player info: {len(players_info):,} players")

    # Convert dates with flexible parsing for mixed formats
    print("Parsing dates...")
    player_stats['gameDate'] = pd.to_datetime(player_stats['gameDate'], format='mixed', errors='coerce')
    players_info['birthdate'] = pd.to_datetime(players_info['birthdate'], format='mixed', errors='coerce')

    print(f"✓ Successfully parsed {player_stats['gameDate'].notna().sum():,} game dates")
    print(f"✓ Successfully parsed {players_info['birthdate'].notna().sum():,} birth dates")

    # Check data quality
    valid_birthdates = players_info[players_info['birthdate'].notna()]
    print(f"✓ Players with valid birth dates: {len(valid_birthdates):,}")

    # Merge datasets
    merged_data = player_stats.merge(
        players_info[['fullName', 'birthdate']],
        on='fullName',
        how='left'
    )

    print(f"✓ Merged dataset: {len(merged_data):,} records")

    return merged_data

def calculate_age_features(data):
    """Calculate age at time of each game and extract year."""

    print("Calculating age features...")

    # Filter to rows with valid dates first
    valid_dates = data[
        (data['gameDate'].notna()) &
        (data['birthdate'].notna())
    ].copy()

    print(f"✓ Records with both valid dates: {len(valid_dates):,}")

    # Extract year from game date for simpler calculation
    valid_dates['game_year'] = valid_dates['gameDate'].dt.year
    valid_dates['birth_year'] = valid_dates['birthdate'].dt.year

    # Calculate approximate age (will be more accurate than exact days for our purposes)
    valid_dates['age_at_game'] = valid_dates['game_year'] - valid_dates['birth_year']

    # Use valid_dates instead of data for the rest of the function
    data = valid_dates

    # Filter to meaningful data
    meaningful_data = data[
        (data['numMinutes'] > 0) &  # Player actually played
        (data['age_at_game'] >= 18) &  # Reasonable age range
        (data['age_at_game'] <= 45) &
        (data['birthdate'].notna()) &  # Has birth date info
        (data['gameDate'].notna()) &  # Valid game date
        (data['points'].notna())  # Has points data
    ].copy()

    print(f"✓ Filtered to {len(meaningful_data):,} meaningful game records")
    print(f"✓ Age range: {meaningful_data['age_at_game'].min():.0f} to {meaningful_data['age_at_game'].max():.0f} years")
    print(f"✓ Unique players: {meaningful_data['fullName'].nunique():,}")

    return meaningful_data

def analyze_age_performance_curves(data):
    """Analyze how performance changes with age across all players."""

    print("\nAnalyzing age-performance relationships...")

    # Group by age and calculate statistics
    age_performance = data.groupby('age_at_game').agg({
        'points': ['count', 'mean', 'std'],
        'numMinutes': 'mean',
        'fieldGoalsPercentage': 'mean',
        'threePointersPercentage': 'mean',
        'freeThrowsPercentage': 'mean'
    }).round(3)

    # Flatten column names
    age_performance.columns = ['_'.join(col).strip() for col in age_performance.columns.values]
    age_performance = age_performance.reset_index()

    # Filter to ages with sufficient data
    age_performance = age_performance[age_performance['points_count'] >= 1000]

    print(f"✓ Age ranges with 1000+ games: {len(age_performance)} ages")
    print(f"✓ Peak scoring age: {age_performance.loc[age_performance['points_mean'].idxmax(), 'age_at_game']}")

    return age_performance

def create_age_based_features(data, age_performance):
    """Create age-based features for our target players."""

    print("\nCreating age-based features...")

    # Create age performance lookup
    age_lookup = age_performance.set_index('age_at_game').to_dict()

    # Add age-based features to data
    data['age_expected_points'] = data['age_at_game'].map(age_lookup['points_mean'])
    data['age_expected_minutes'] = data['age_at_game'].map(age_lookup['numMinutes_mean'])
    data['age_expected_fg_pct'] = data['age_at_game'].map(age_lookup['fieldGoalsPercentage_mean'])

    # Create relative performance features
    data['points_vs_age_expected'] = data['points'] - data['age_expected_points']
    data['minutes_vs_age_expected'] = data['numMinutes'] - data['age_expected_minutes']

    # Age categories
    data['age_category'] = pd.cut(data['age_at_game'],
                                 bins=[0, 22, 27, 32, 50],
                                 labels=['Young', 'Prime', 'Veteran', 'Old'])

    # Career stage (years in league approximation)
    data['years_in_league'] = data['age_at_game'] - 19  # Assume entered league at 19
    data['career_stage'] = pd.cut(data['years_in_league'],
                                 bins=[0, 3, 8, 12, 30],
                                 labels=['Rookie', 'Developing', 'Peak', 'Veteran'])

    print(f"✓ Added age-based features")

    return data

def analyze_target_players_age_patterns(data):
    """Analyze age patterns specifically for our 13 target players."""

    # Our target players with corrected names
    target_players = {
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

    print("\nAnalyzing target players' age patterns...")

    target_data = data[data['fullName'].isin(target_players.values())].copy()

    # Current age analysis (2024-25 season)
    current_season = target_data[target_data['game_year'] >= 2024]

    if len(current_season) > 0:
        player_current_age = current_season.groupby('fullName').agg({
            'age_at_game': 'mean',
            'points': 'mean',
            'age_expected_points': 'mean',
            'points_vs_age_expected': 'mean',
            'age_category': lambda x: x.mode()[0] if not x.empty else 'Unknown'
        }).round(2)

        print(f"✓ Current season age analysis for {len(player_current_age)} players:")
        print(player_current_age)

        return player_current_age
    else:
        print("⚠️ No current season data found")
        return None

def save_age_analysis_results(age_performance, target_age_analysis):
    """Save the age analysis results for use in feature engineering."""

    print("\nSaving age analysis results...")

    # Save age performance curves
    age_performance.to_csv('../data/processed/age_performance_curves.csv', index=False)
    print("✓ Saved age performance curves to ../data/processed/age_performance_curves.csv")

    # Save target player age analysis
    if target_age_analysis is not None:
        target_age_analysis.to_csv('../data/processed/target_players_age_analysis.csv')
        print("✓ Saved target player age analysis to ../data/processed/target_players_age_analysis.csv")

    return True

def main():
    """Main analysis workflow."""

    # Step 1: Load and merge data
    merged_data = load_and_merge_data()

    # Step 2: Calculate age features
    meaningful_data = calculate_age_features(merged_data)

    # Step 3: Analyze age-performance curves across all players
    age_performance = analyze_age_performance_curves(meaningful_data)

    # Step 4: Create age-based features
    feature_data = create_age_based_features(meaningful_data, age_performance)

    # Step 5: Analyze target players specifically
    target_analysis = analyze_target_players_age_patterns(feature_data)

    # Step 6: Save results
    save_age_analysis_results(age_performance, target_analysis)

    print("\n=== AGE ANALYSIS COMPLETE ===")
    print("✅ Analyzed age-performance patterns across all NBA players")
    print("✅ Created age-based features for prediction models")
    print("✅ Analyzed current age status of target players")
    print("✅ Saved results for feature engineering pipeline")

    return feature_data, age_performance, target_analysis

if __name__ == "__main__":
    main()