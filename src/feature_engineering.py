"""
NBA Feature Engineering Pipeline

This module creates advanced features for our over/under prediction models including:
- Rolling averages (3, 5, 10 games)
- Rest days and fatigue indicators
- Opponent strength metrics
- Streak and momentum features
- Contextual features (home/away, season progression)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_cleaned_data():
    """Load the cleaned data and supporting datasets."""

    print("=== NBA FEATURE ENGINEERING PIPELINE ===")
    print("Loading cleaned datasets...")

    # Load main cleaned data
    cleaned_data = pd.read_csv('../data/processed/cleaned_player_data.csv')

    # Load supporting data
    games = pd.read_csv('../data/raw/Games.csv')
    team_stats = pd.read_csv('../data/raw/TeamStatistics.csv')

    print(f"✓ Cleaned player data: {len(cleaned_data):,} games")
    print(f"✓ Games dataset: {len(games):,} games")
    print(f"✓ Team stats: {len(team_stats):,} records")

    return cleaned_data, games, team_stats

def create_rolling_features(data):
    """Create rolling average features for each player."""

    print("\nCreating rolling average features...")

    # Ensure data is sorted by player and date
    data = data.sort_values(['fullName', 'gameDate']).copy()

    # Initialize rolling feature columns
    rolling_windows = [3, 5, 10]
    rolling_features = [
        'points', 'numMinutes', 'fieldGoalsMade', 'fieldGoalsAttempted',
        'threePointersMade', 'threePointersAttempted', 'freeThrowsMade',
        'freeThrowsAttempted', 'assists', 'reboundsTotal', 'steals', 'blocks'
    ]

    feature_count = 0

    for window in rolling_windows:
        print(f"  - Creating {window}-game rolling averages...")

        for feature in rolling_features:
            if feature in data.columns:
                col_name = f'rolling_{window}g_{feature}'

                # Create rolling average (excluding current game)
                data[col_name] = data.groupby('fullName')[feature].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
                feature_count += 1

        # Rolling shooting percentages
        data[f'rolling_{window}g_fg_pct'] = data.groupby('fullName').apply(
            lambda group: (
                group['fieldGoalsMade'].rolling(window=window, min_periods=1).sum().shift(1) /
                group['fieldGoalsAttempted'].rolling(window=window, min_periods=1).sum().shift(1)
            ).fillna(0)
        ).values

        data[f'rolling_{window}g_3pt_pct'] = data.groupby('fullName').apply(
            lambda group: (
                group['threePointersMade'].rolling(window=window, min_periods=1).sum().shift(1) /
                group['threePointersAttempted'].rolling(window=window, min_periods=1).sum().shift(1)
            ).fillna(0)
        ).values

        feature_count += 2

    print(f"✓ Created {feature_count} rolling average features")

    return data

def create_rest_and_fatigue_features(data):
    """Create rest days and fatigue-related features."""

    print("\nCreating rest and fatigue features...")

    # Ensure gameDate is datetime
    data['gameDate'] = pd.to_datetime(data['gameDate'])

    # Sort by player and date
    data = data.sort_values(['fullName', 'gameDate']).copy()

    # Calculate days since last game
    data['days_rest'] = data.groupby('fullName')['gameDate'].diff().dt.days
    data['days_rest'] = data['days_rest'].fillna(3)  # Assume 3 days rest for first game

    # Rest categories
    data['rest_category'] = pd.cut(
        data['days_rest'],
        bins=[0, 1, 2, 3, 30],
        labels=['Back_to_Back', 'One_Day', 'Two_Days', 'Well_Rested'],
        include_lowest=True
    )

    # Games in last N days (fatigue proxy) - use game-based approximation
    data['games_last_7d'] = data.groupby('fullName')['gameDate'].transform(
        lambda x: x.rolling(window=4, min_periods=1).count().shift(1)  # Approximate 7 days as 4 games
    ).fillna(1)

    data['games_last_14d'] = data.groupby('fullName')['gameDate'].transform(
        lambda x: x.rolling(window=7, min_periods=1).count().shift(1)  # Approximate 14 days as 7 games
    ).fillna(2)

    # Minutes load (fatigue indicator)
    for window in [3, 7]:
        data[f'avg_minutes_last_{window}g'] = data.groupby('fullName')['numMinutes'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )

    # Back-to-back indicator
    data['is_back_to_back'] = (data['days_rest'] <= 1).astype(int)

    print("✓ Created rest and fatigue features")

    return data

def create_momentum_and_streak_features(data):
    """Create streak and momentum features."""

    print("\nCreating momentum and streak features...")

    # Sort by player and date
    data = data.sort_values(['fullName', 'gameDate']).copy()

    # Points vs threshold (for momentum)
    data['points_vs_threshold'] = data['points'] - data['player_threshold']

    # Create streak features
    for window in [3, 5]:
        # Over/under streaks
        data[f'over_streak_last_{window}g'] = data.groupby('fullName')['over_threshold'].transform(
            lambda x: x.rolling(window=window, min_periods=1).sum().shift(1)
        )

        # Performance vs threshold trend
        data[f'vs_threshold_trend_{window}g'] = data.groupby('fullName')['points_vs_threshold'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )

        # Minutes trend (usage trend)
        data[f'minutes_trend_{window}g'] = data.groupby('fullName')['numMinutes'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )

    # Hot/cold streak indicators
    data['hot_streak'] = (data['over_streak_last_5g'] >= 4).astype(int)
    data['cold_streak'] = (data['over_streak_last_5g'] <= 1).astype(int)

    # Performance relative to recent average
    data['points_vs_rolling_avg'] = data['points'] - data['rolling_10g_points']

    print("✓ Created momentum and streak features")

    return data

def create_contextual_features(data, games):
    """Create contextual features about game situation."""

    print("\nCreating contextual features...")

    # Merge with games data for additional context
    games['gameDate'] = pd.to_datetime(games['gameDate'], format='mixed', errors='coerce')

    # Create game mapping
    game_context = games[['gameId', 'gameDate', 'hometeamCity', 'awayteamCity']].copy() if 'gameId' in games.columns else None

    # Season progression (days since season start)
    season_start = data['gameDate'].min()
    data['days_since_season_start'] = (data['gameDate'] - season_start).dt.days

    # Season progression as percentage (assume 185 day season)
    data['season_progression'] = np.minimum(data['days_since_season_start'] / 185, 1.0)

    # Month of season
    data['month'] = data['gameDate'].dt.month
    data['season_month'] = pd.cut(
        data['season_progression'],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=['Early', 'Mid_Early', 'Mid_Late', 'Late'],
        include_lowest=True
    )

    # Day of week effects
    data['day_of_week'] = data['gameDate'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

    # Home/away encoding (if available)
    if 'home' in data.columns:
        data['home_game'] = data['home'].astype(int)
    else:
        data['home_game'] = 0  # Default if not available

    print("✓ Created contextual features")

    return data

def create_opponent_features(data, team_stats):
    """Create opponent strength features."""

    print("\nCreating opponent features...")

    # This is a simplified version since opponent matching is complex
    # In a full implementation, you would match games to opponents and their stats

    # For now, create placeholder features that can be filled in later
    # when opponent data is properly matched

    # Average opponent strength (placeholder - could be calculated from actual matchups)
    data['opponent_def_rating'] = 100.0  # League average placeholder
    data['opponent_pace'] = 100.0  # League average placeholder

    # Recent performance vs similar opponents (placeholder)
    data['vs_strong_def_last_5g'] = 0.5  # Placeholder

    print("✓ Created opponent features (simplified)")

    return data

def create_advanced_features(data):
    """Create advanced statistical features."""

    print("\nCreating advanced features...")

    # Usage rate approximation (based on shots and minutes)
    data['usage_rate_approx'] = np.where(
        data['numMinutes'] > 0,
        (data['fieldGoalsAttempted'] + 0.44 * data['freeThrowsAttempted']) / data['numMinutes'],
        0
    )

    # Efficiency metrics
    data['points_per_shot'] = np.where(
        data['total_shot_attempts'] > 0,
        data['points'] / data['total_shot_attempts'],
        0
    )

    # Performance consistency (rolling standard deviation)
    for window in [5, 10]:
        data[f'points_volatility_{window}g'] = data.groupby('fullName')['points'].transform(
            lambda x: x.rolling(window=window, min_periods=2).std().shift(1)
        ).fillna(0)

    # Hot hand indicators (recent shooting performance)
    data['recent_fg_hot'] = (data['rolling_3g_fg_pct'] > data['rolling_10g_fg_pct']).astype(int)
    data['recent_3pt_hot'] = (data['rolling_3g_3pt_pct'] > data['rolling_10g_3pt_pct']).astype(int)

    # Form indicators
    data['good_form'] = (data['vs_threshold_trend_5g'] > 0).astype(int)
    data['high_usage'] = (data['usage_rate_approx'] > data.groupby('fullName')['usage_rate_approx'].transform('median')).astype(int)

    print("✓ Created advanced features")

    return data

def clean_and_validate_features(data):
    """Clean and validate all engineered features."""

    print("\nCleaning and validating features...")

    original_shape = data.shape

    # Handle infinite values
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)

    # Fill NaN values with appropriate defaults
    fill_values = {
        'days_rest': 3,
        'games_last_7d': 1,
        'games_last_14d': 2,
        'is_back_to_back': 0,
        'hot_streak': 0,
        'cold_streak': 0,
        'home_game': 0,
        'is_weekend': 0
    }

    for col, fill_val in fill_values.items():
        if col in data.columns:
            data[col] = data[col].fillna(fill_val)

    # Fill remaining NaN values in rolling features with 0
    rolling_cols = [col for col in data.columns if 'rolling_' in col or 'trend_' in col or 'avg_' in col]
    for col in rolling_cols:
        data[col] = data[col].fillna(0)

    # Ensure categorical features are properly encoded
    categorical_cols = ['rest_category', 'season_month']
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype(str)

    print(f"✓ Data validation complete")
    print(f"  - Shape: {original_shape} -> {data.shape}")
    print(f"  - NaN values remaining: {data.isnull().sum().sum()}")

    return data

def save_engineered_features(data):
    """Save the engineered features dataset."""

    print("\nSaving engineered features...")

    # Ensure directory exists
    os.makedirs('../data/processed', exist_ok=True)

    # Save engineered dataset
    output_file = '../data/processed/engineered_features.csv'
    data.to_csv(output_file, index=False)

    print(f"✓ Saved engineered features: {output_file}")
    print(f"  - Shape: {data.shape}")
    print(f"  - Memory usage: {data.memory_usage(deep=True).sum() / (1024**2):.1f} MB")

    # Create feature summary
    feature_summary = {
        'total_features': len(data.columns),
        'total_games': len(data),
        'players': data['fullName'].nunique(),
        'date_range_start': str(data['gameDate'].min()),
        'date_range_end': str(data['gameDate'].max()),
        'over_rate': data['over_threshold'].mean(),
        'feature_engineering_timestamp': datetime.now().isoformat()
    }

    # Save feature list and types
    feature_info = pd.DataFrame({
        'feature_name': data.columns,
        'dtype': [str(dtype) for dtype in data.dtypes],
        'null_count': data.isnull().sum().values,
        'unique_values': [data[col].nunique() if data[col].dtype != 'object' else 'categorical' for col in data.columns]
    })

    feature_info.to_csv('../data/processed/feature_summary.csv', index=False)

    summary_df = pd.DataFrame([feature_summary])
    summary_df.to_csv('../data/processed/feature_engineering_summary.csv', index=False)

    print("✓ Saved feature summaries")

    return feature_summary

def main():
    """Main feature engineering workflow."""

    # Step 1: Load cleaned data
    cleaned_data, games, team_stats = load_cleaned_data()

    # Step 2: Create rolling average features
    with_rolling = create_rolling_features(cleaned_data)

    # Step 3: Create rest and fatigue features
    with_rest = create_rest_and_fatigue_features(with_rolling)

    # Step 4: Create momentum and streak features
    with_momentum = create_momentum_and_streak_features(with_rest)

    # Step 5: Create contextual features
    with_context = create_contextual_features(with_momentum, games)

    # Step 6: Create opponent features (simplified)
    with_opponents = create_opponent_features(with_context, team_stats)

    # Step 7: Create advanced features
    with_advanced = create_advanced_features(with_opponents)

    # Step 8: Clean and validate
    final_features = clean_and_validate_features(with_advanced)

    # Step 9: Save engineered features
    summary = save_engineered_features(final_features)

    print("\n=== FEATURE ENGINEERING COMPLETE ===")
    print(f"✅ Created {summary['total_features']} features for {summary['total_games']:,} games")
    print(f"✅ {summary['players']} players with rolling averages, momentum, and contextual features")
    print(f"✅ Over rate: {summary['over_rate']:.1%} (balanced)")
    print(f"✅ Ready for baseline models and ML training")

    return final_features

if __name__ == "__main__":
    main()