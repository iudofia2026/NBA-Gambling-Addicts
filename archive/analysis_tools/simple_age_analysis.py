"""
Simplified NBA Age Analysis - Working Version
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=== SIMPLIFIED NBA AGE ANALYSIS ===")

    # Load data
    print("Loading data...")
    player_stats = pd.read_csv('../data/raw/PlayerStatistics.csv', low_memory=False)
    players_info = pd.read_csv('../data/raw/Players.csv')

    # Create names
    player_stats['fullName'] = player_stats['firstName'].astype(str) + ' ' + player_stats['lastName'].astype(str)
    players_info['fullName'] = players_info['firstName'].astype(str) + ' ' + players_info['lastName'].astype(str)

    print(f"✓ Loaded {len(player_stats):,} game records")
    print(f"✓ Loaded {len(players_info):,} player profiles")

    # Simple approach: extract year directly from string
    print("Extracting years from date strings...")

    # Extract year from game date (handle multiple formats)
    def extract_year(date_str):
        try:
            if pd.isna(date_str):
                return None
            date_str = str(date_str)
            # Most dates start with year
            return int(date_str[:4])
        except:
            return None

    player_stats['game_year'] = player_stats['gameDate'].apply(extract_year)

    # Extract birth year from birthdate
    def extract_birth_year(date_str):
        try:
            if pd.isna(date_str):
                return None
            date_str = str(date_str)
            # Birth dates are in YYYY-MM-DD format
            return int(date_str[:4])
        except:
            return None

    players_info['birth_year'] = players_info['birthdate'].apply(extract_birth_year)

    # Merge data
    merged_data = player_stats.merge(
        players_info[['fullName', 'birth_year']],
        on='fullName',
        how='left'
    )

    # Calculate age
    merged_data['age_at_game'] = merged_data['game_year'] - merged_data['birth_year']

    # Filter meaningful data
    meaningful_data = merged_data[
        (merged_data['numMinutes'] > 0) &
        (merged_data['age_at_game'] >= 18) &
        (merged_data['age_at_game'] <= 45) &
        (merged_data['birth_year'].notna()) &
        (merged_data['game_year'].notna())
    ]

    print(f"✓ Meaningful records: {len(meaningful_data):,}")
    print(f"✓ Age range: {meaningful_data['age_at_game'].min()} to {meaningful_data['age_at_game'].max()}")
    print(f"✓ Unique players: {meaningful_data['fullName'].nunique():,}")
    print()

    # Analyze age patterns
    print("ANALYZING AGE-PERFORMANCE PATTERNS")
    print("=" * 40)

    age_stats = meaningful_data.groupby('age_at_game').agg({
        'points': ['count', 'mean', 'std'],
        'numMinutes': 'mean'
    }).round(2)

    # Flatten columns
    age_stats.columns = ['games', 'avg_points', 'std_points', 'avg_minutes']
    age_stats = age_stats.reset_index()

    # Filter to ages with enough data
    age_stats = age_stats[age_stats['games'] >= 1000]

    print("Age-based performance (ages with 1000+ games):")
    print(age_stats)

    # Find peak performance age
    peak_age = age_stats.loc[age_stats['avg_points'].idxmax(), 'age_at_game']
    peak_points = age_stats.loc[age_stats['avg_points'].idxmax(), 'avg_points']

    print(f"\\nPeak scoring age: {peak_age} years ({peak_points:.1f} avg points)")
    print()

    # Analyze our target players
    target_players = [
        'Mikal Bridges', 'Buddy Hield', 'Harrison Barnes', 'Nikola Jokic',
        'James Harden', 'Rudy Gobert', 'Nikola Vucevic', 'Tobias Harris',
        'Devin Booker', 'Karl-Anthony Towns', 'Jrue Holiday', 'Stephen Curry', 'Kevin Durant'
    ]

    print("TARGET PLAYER AGE ANALYSIS (2024-25 season)")
    print("=" * 50)

    # Recent season data
    recent_data = meaningful_data[meaningful_data['game_year'] >= 2024]
    target_recent = recent_data[recent_data['fullName'].isin(target_players)]

    print(f"Recent games found: {len(target_recent)} from {len(target_recent['fullName'].unique()) if len(target_recent) > 0 else 0} players")

    if len(target_recent) > 0:
        target_analysis = target_recent.groupby('fullName').agg({
            'age_at_game': 'mean',
            'points': 'mean',
            'numMinutes': 'mean'
        }).round(1)

        # Add expected points based on age
        target_analysis = target_analysis.reset_index()

        # Merge with age expectations
        age_lookup = age_stats.set_index('age_at_game')['avg_points'].to_dict()
        target_analysis['expected_points_for_age'] = target_analysis['age_at_game'].round().map(age_lookup)
        target_analysis['points_vs_age_expected'] = target_analysis['points'] - target_analysis['expected_points_for_age']

        print(target_analysis)
        print()

        # Save results
        print("Saving results...")
        import os
        os.makedirs('../data/processed', exist_ok=True)

        age_stats.to_csv('../data/processed/age_performance_curves.csv', index=False)
        target_analysis.to_csv('../data/processed/target_players_age_analysis.csv', index=False)

        print("✓ Saved age_performance_curves.csv")
        print("✓ Saved target_players_age_analysis.csv")

    # Summary insights
    print()
    print("KEY INSIGHTS:")
    print("=" * 15)
    print(f"• Peak NBA scoring age: {peak_age} years")
    print(f"• Decline after peak: {age_stats[age_stats['age_at_game'] > peak_age]['avg_points'].iloc[0] - peak_points:.1f} points at age {peak_age + 1}")

    # Age categories
    young_avg = age_stats[age_stats['age_at_game'] <= 23]['avg_points'].mean()
    prime_avg = age_stats[(age_stats['age_at_game'] > 23) & (age_stats['age_at_game'] <= 30)]['avg_points'].mean()
    veteran_avg = age_stats[age_stats['age_at_game'] > 30]['avg_points'].mean()

    print(f"• Young players (≤23): {young_avg:.1f} avg points")
    print(f"• Prime players (24-30): {prime_avg:.1f} avg points")
    print(f"• Veterans (30+): {veteran_avg:.1f} avg points")

    print()
    print("✅ AGE ANALYSIS COMPLETE")
    print("✅ Ready to incorporate age factors into prediction models")

if __name__ == "__main__":
    main()