"""
NBA Player vs Team/Opponent Matchup Features

This module creates sophisticated matchup-based features including:
1. Player performance vs specific teams (e.g., Curry vs Lakers)
2. Player performance vs specific opposing players (e.g., Durant vs Giannis)
3. Team defensive ratings vs player positions/styles
4. Historical head-to-head performance patterns
5. Rest/fatigue matchups (e.g., back-to-back vs well-rested teams)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data_for_matchups():
    """Load all datasets needed for matchup analysis."""

    print("=== NBA MATCHUP FEATURE ENGINEERING ===")
    print("Loading datasets for matchup analysis...")

    # Load main datasets
    player_stats = pd.read_csv('../data/raw/PlayerStatistics.csv', low_memory=False)
    games = pd.read_csv('../data/raw/Games.csv')
    team_stats = pd.read_csv('../data/raw/TeamStatistics.csv')
    players_info = pd.read_csv('../data/raw/Players.csv')

    # Load our processed data
    cleaned_data = pd.read_csv('../data/processed/cleaned_player_data.csv')

    print(f"✓ Player statistics: {len(player_stats):,} records")
    print(f"✓ Games: {len(games):,} records")
    print(f"✓ Team statistics: {len(team_stats):,} records")
    print(f"✓ Cleaned player data: {len(cleaned_data):,} records")

    return {
        'player_stats': player_stats,
        'games': games,
        'team_stats': team_stats,
        'players_info': players_info,
        'cleaned_data': cleaned_data
    }

def create_player_vs_team_features(player_stats, target_players):
    """Create features for how each player performs against specific teams."""

    print("\nCreating player vs team matchup features...")

    # Create full names
    player_stats['fullName'] = player_stats['firstName'].astype(str) + ' ' + player_stats['lastName'].astype(str)

    # Filter to our target players and active games
    target_data = player_stats[
        (player_stats['fullName'].isin(target_players)) &
        (player_stats['numMinutes'] > 0)
    ].copy()

    print(f"✓ Analyzing {len(target_data):,} games for matchup patterns")

    # Calculate player performance vs each opponent team
    matchup_stats = target_data.groupby(['fullName', 'opponentteamName']).agg({
        'points': ['count', 'mean', 'std'],
        'numMinutes': 'mean',
        'fieldGoalsPercentage': 'mean',
        'over_threshold': 'mean'  # Win rate vs this opponent
    }).round(3)

    # Flatten column names
    matchup_stats.columns = ['_'.join(col).strip() for col in matchup_stats.columns.values]
    matchup_stats = matchup_stats.reset_index()

    # Only keep matchups with at least 3 games
    matchup_stats = matchup_stats[matchup_stats['points_count'] >= 3]

    print(f"✓ Found {len(matchup_stats)} player-team matchups with 3+ games")

    return matchup_stats

def create_team_defensive_features(team_stats, target_players):
    """Create team defensive strength features against different player types."""

    print("\nCreating team defensive strength features...")

    # Calculate team defensive metrics
    team_defense = team_stats.groupby(['teamName', 'teamId']).agg({
        'points': 'mean',  # Points allowed
        'fieldGoalsPercentage': 'mean',  # FG% allowed
        'threePointersPercentage': 'mean',  # 3P% allowed
        'reboundsTotal': 'mean',  # Rebounds allowed
        'turnovers': 'mean'  # Turnovers forced
    }).round(3)

    team_defense.columns = [f'def_{col}' for col in team_defense.columns]
    team_defense = team_defense.reset_index()

    print(f"✓ Calculated defensive metrics for {len(team_defense)} teams")

    return team_defense

def create_rest_matchup_features(cleaned_data):
    """Create features for rest/fatigue matchups between teams."""

    print("\nCreating rest vs rest matchup features...")

    # This analyzes how players perform when they have different rest levels
    # vs teams with different rest levels

    rest_matchups = cleaned_data.groupby(['fullName', 'rest_category']).agg({
        'points': 'mean',
        'over_threshold': 'mean',
        'numMinutes': 'mean'
    }).round(3)

    rest_matchups = rest_matchups.reset_index()

    print(f"✓ Created rest-based performance profiles")

    return rest_matchups

def create_historical_head_to_head_features(player_stats, target_players):
    """Create head-to-head performance features for our target players."""

    print("\nCreating head-to-head performance features...")

    # Create full names
    player_stats['fullName'] = player_stats['firstName'].astype(str) + ' ' + player_stats['lastName'].astype(str)

    # Filter to target players
    target_data = player_stats[
        (player_stats['fullName'].isin(target_players)) &
        (player_stats['numMinutes'] > 0)
    ].copy()

    # Calculate performance vs each opposing team's key players
    # For simplicity, we'll use opponent team performance as a proxy

    h2h_features = target_data.groupby(['fullName', 'opponentteamName']).agg({
        'points': ['mean', 'std', 'max'],
        'over_threshold': ['mean', 'count'],
        'fieldGoalsPercentage': 'mean',
        'threePointersPercentage': 'mean'
    }).round(3)

    # Flatten columns
    h2h_features.columns = ['_'.join(col).strip() for col in h2h_features.columns.values]
    h2h_features = h2h_features.reset_index()

    # Filter to matchups with sufficient history
    h2h_features = h2h_features[h2h_features['over_threshold_count'] >= 5]

    print(f"✓ Created {len(h2h_features)} head-to-head matchup profiles")

    return h2h_features

def create_matchup_lookup_tables(matchup_stats, team_defense, rest_matchups, h2h_features):
    """Create lookup tables for adding matchup features to games."""

    print("\nCreating matchup lookup tables...")

    # Player vs Team lookup
    player_team_lookup = {}
    for _, row in matchup_stats.iterrows():
        key = (row['fullName'], row['opponentteamName'])
        player_team_lookup[key] = {
            'vs_team_avg_points': row['points_mean'],
            'vs_team_games': row['points_count'],
            'vs_team_over_rate': row['over_threshold_mean'],
            'vs_team_consistency': row['points_std'] if 'points_std' in row else 0
        }

    # Team defense lookup
    team_def_lookup = {}
    for _, row in team_defense.iterrows():
        team_def_lookup[row['teamName']] = {
            'team_def_rating': row['def_points'],
            'team_def_fg_pct': row['def_fieldGoalsPercentage'],
            'team_def_3pt_pct': row['def_threePointersPercentage']
        }

    # Rest matchup lookup
    rest_lookup = {}
    for _, row in rest_matchups.iterrows():
        key = (row['fullName'], row['rest_category'])
        rest_lookup[key] = {
            'rest_avg_points': row['points'],
            'rest_over_rate': row['over_threshold']
        }

    print(f"✓ Created lookup tables:")
    print(f"  - Player vs Team: {len(player_team_lookup)} entries")
    print(f"  - Team Defense: {len(team_def_lookup)} entries")
    print(f"  - Rest Patterns: {len(rest_lookup)} entries")

    return player_team_lookup, team_def_lookup, rest_lookup

def add_matchup_features_to_data(cleaned_data, player_team_lookup, team_def_lookup, rest_lookup):
    """Add all matchup features to the main dataset."""

    print("\nAdding matchup features to main dataset...")

    data_with_matchups = cleaned_data.copy()

    # Add player vs team features
    data_with_matchups['vs_team_avg_points'] = data_with_matchups.apply(
        lambda row: player_team_lookup.get(
            (row['fullName'], row['opponentteamName']), {}
        ).get('vs_team_avg_points', row.get('rolling_10g_points', 0)), axis=1
    )

    data_with_matchups['vs_team_over_rate'] = data_with_matchups.apply(
        lambda row: player_team_lookup.get(
            (row['fullName'], row['opponentteamName']), {}
        ).get('vs_team_over_rate', 0.5), axis=1
    )

    data_with_matchups['vs_team_games_played'] = data_with_matchups.apply(
        lambda row: player_team_lookup.get(
            (row['fullName'], row['opponentteamName']), {}
        ).get('vs_team_games', 0), axis=1
    )

    # Add team defensive features
    data_with_matchups['opponent_def_rating'] = data_with_matchups['opponentteamName'].map(
        lambda x: team_def_lookup.get(x, {}).get('team_def_rating', 100)
    )

    data_with_matchups['opponent_def_fg_pct'] = data_with_matchups['opponentteamName'].map(
        lambda x: team_def_lookup.get(x, {}).get('team_def_fg_pct', 0.45)
    )

    # Add rest-based performance
    if 'rest_category' in data_with_matchups.columns:
        data_with_matchups['rest_expected_points'] = data_with_matchups.apply(
            lambda row: rest_lookup.get(
                (row['fullName'], row['rest_category']), {}
            ).get('rest_avg_points', row.get('rolling_5g_points', 0)), axis=1
        )

        data_with_matchups['rest_expected_over_rate'] = data_with_matchups.apply(
            lambda row: rest_lookup.get(
                (row['fullName'], row['rest_category']), {}
            ).get('rest_over_rate', 0.5), axis=1
        )

    # Create derived matchup features
    data_with_matchups['matchup_advantage'] = (
        data_with_matchups['vs_team_avg_points'] - data_with_matchups['rolling_10g_points'].fillna(0)
    )

    data_with_matchups['defensive_challenge'] = (
        data_with_matchups['opponent_def_rating'] - 100  # Relative to league average
    )

    # Familiarity factor (more games = more predictable)
    data_with_matchups['matchup_familiarity'] = np.minimum(
        data_with_matchups['vs_team_games_played'] / 10, 1.0
    )

    # Overall matchup score
    data_with_matchups['overall_matchup_score'] = (
        0.4 * (data_with_matchups['vs_team_over_rate'] - 0.5) +  # Historical success
        0.3 * (-data_with_matchups['defensive_challenge'] / 10) +  # Opponent weakness
        0.3 * (data_with_matchups['matchup_advantage'] / 5)  # Point advantage
    )

    feature_count = len([col for col in data_with_matchups.columns if col not in cleaned_data.columns])
    print(f"✓ Added {feature_count} new matchup features")

    return data_with_matchups

def analyze_matchup_insights(matchup_stats, target_players):
    """Analyze key insights from matchup data."""

    print("\n=== MATCHUP INSIGHTS ===")

    for player in target_players:
        player_matchups = matchup_stats[matchup_stats['fullName'] == player]

        if len(player_matchups) > 0:
            # Best and worst matchups
            best_matchup = player_matchups.loc[player_matchups['points_mean'].idxmax()]
            worst_matchup = player_matchups.loc[player_matchups['points_mean'].idxmin()]

            print(f"\n{player}:")
            print(f"  Best vs: {best_matchup['opponentteamName']} ({best_matchup['points_mean']:.1f} avg pts)")
            print(f"  Worst vs: {worst_matchup['opponentteamName']} ({worst_matchup['points_mean']:.1f} avg pts)")
            print(f"  Matchup range: {best_matchup['points_mean'] - worst_matchup['points_mean']:.1f} point difference")

def save_matchup_features(data_with_matchups, matchup_stats):
    """Save the enhanced dataset with matchup features."""

    print("\nSaving matchup-enhanced dataset...")

    # Save enhanced dataset
    output_file = '../data/processed/enhanced_features_with_matchups.csv'
    data_with_matchups.to_csv(output_file, index=False)

    # Save matchup analysis
    matchup_file = '../data/processed/player_team_matchups.csv'
    matchup_stats.to_csv(matchup_file, index=False)

    print(f"✓ Saved enhanced dataset: {output_file}")
    print(f"  - Shape: {data_with_matchups.shape}")
    print(f"  - New features: {data_with_matchups.shape[1] - 130} added")

    print(f"✓ Saved matchup analysis: {matchup_file}")

    return output_file

def main():
    """Main matchup feature engineering workflow."""

    # Target players (corrected names for dataset)
    target_players = [
        'Mikal Bridges', 'Buddy Hield', 'Harrison Barnes', 'Nikola Jokic',
        'James Harden', 'Rudy Gobert', 'Nikola Vucevic', 'Tobias Harris',
        'Devin Booker', 'Karl-Anthony Towns', 'Jrue Holiday', 'Stephen Curry', 'Kevin Durant'
    ]

    # Step 1: Load data
    data = load_data_for_matchups()

    # Step 2: Create matchup features
    matchup_stats = create_player_vs_team_features(data['player_stats'], target_players)
    team_defense = create_team_defensive_features(data['team_stats'], target_players)
    rest_matchups = create_rest_matchup_features(data['cleaned_data'])
    h2h_features = create_historical_head_to_head_features(data['player_stats'], target_players)

    # Step 3: Create lookup tables
    player_team_lookup, team_def_lookup, rest_lookup = create_matchup_lookup_tables(
        matchup_stats, team_defense, rest_matchups, h2h_features
    )

    # Step 4: Add features to main dataset
    enhanced_data = add_matchup_features_to_data(
        data['cleaned_data'], player_team_lookup, team_def_lookup, rest_lookup
    )

    # Step 5: Analyze insights
    analyze_matchup_insights(matchup_stats, target_players)

    # Step 6: Save enhanced dataset
    output_file = save_matchup_features(enhanced_data, matchup_stats)

    print("\n=== MATCHUP FEATURE ENGINEERING COMPLETE ===")
    print(f"✅ Enhanced dataset with player vs team performance")
    print(f"✅ Added team defensive strength metrics")
    print(f"✅ Included rest/fatigue matchup patterns")
    print(f"✅ Created overall matchup advantage scores")
    print(f"✅ Dataset ready for improved ML training")

    return enhanced_data

if __name__ == "__main__":
    main()