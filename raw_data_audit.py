#!/usr/bin/env python3
"""
COMPREHENSIVE RAW DATA AUDIT
Identifies all data utilization issues affecting prediction quality
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def audit_raw_data():
    """Comprehensive audit of raw data utilization issues."""

    print("🔍 COMPREHENSIVE RAW DATA AUDIT")
    print("=" * 80)

    try:
        # Load data
        data = pd.read_csv('data/processed/engineered_features.csv')
        print(f"✅ Loaded {len(data):,} records")

        print(f"\n📊 BASIC DATA QUALITY AUDIT")
        print("=" * 50)

        # 1. Date Analysis
        print("\n🗓️  DATE UTILIZATION ANALYSIS:")
        if 'gameDate' in data.columns:
            data['gameDate_parsed'] = pd.to_datetime(data['gameDate'], errors='coerce')

            # Check date range
            min_date = data['gameDate_parsed'].min()
            max_date = data['gameDate_parsed'].max()
            print(f"   Date Range: {min_date} to {max_date}")

            # How recent is the data?
            today = pd.Timestamp.now().tz_localize('UTC') if max_date.tz else pd.Timestamp.now()
            days_since_last = (today - max_date).days
            print(f"   Days Since Last Game: {days_since_last} days")

            if days_since_last > 30:
                print(f"   🚨 DATA IS STALE! Last update {days_since_last} days ago")
            else:
                print(f"   ✅ Data is recent (within 30 days)")

            # Date parsing issues
            null_dates = data['gameDate_parsed'].isnull().sum()
            print(f"   Unparseable dates: {null_dates:,} ({null_dates/len(data)*100:.1f}%)")

        else:
            print("   ❌ NO DATE COLUMN FOUND!")

        # 2. Player Coverage Analysis
        print("\n🏀 PLAYER COVERAGE ANALYSIS:")
        unique_players = data['fullName'].nunique() if 'fullName' in data.columns else 0
        print(f"   Unique Players: {unique_players:,}")

        # Sample star players to check coverage
        star_players = ['LeBron James', 'Stephen Curry', 'Kevin Durant', 'Nikola Jokic', 'Giannis Antetokounmpo']
        missing_stars = []
        for player in star_players:
            if player not in data['fullName'].values:
                missing_stars.append(player)

        if missing_stars:
            print(f"   🚨 Missing Star Players: {missing_stars}")
        else:
            print(f"   ✅ All major stars present in dataset")

        # 3. Statistical Completeness
        print(f"\n📈 STATISTICAL COMPLETENESS:")
        key_stats = ['points', 'reboundsTotal', 'assists', 'numMinutes']

        for stat in key_stats:
            if stat in data.columns:
                null_count = data[stat].isnull().sum()
                zero_count = (data[stat] == 0).sum()
                print(f"   {stat:15s}: {null_count:6,} nulls ({null_count/len(data)*100:4.1f}%), {zero_count:6,} zeros ({zero_count/len(data)*100:4.1f}%)")
            else:
                print(f"   {stat:15s}: ❌ COLUMN MISSING")

        # 4. Minutes Analysis (Critical for starter identification)
        print(f"\n⏱️  MINUTES ANALYSIS:")
        if 'numMinutes' in data.columns:
            minutes = data['numMinutes'].dropna()
            print(f"   Average Minutes: {minutes.mean():.1f}")
            print(f"   Starter Games (28+ min): {(minutes >= 28).sum():,} ({(minutes >= 28).sum()/len(minutes)*100:.1f}%)")
            print(f"   Bench Games (<15 min): {(minutes < 15).sum():,} ({(minutes < 15).sum()/len(minutes)*100:.1f}%)")

            # This is critical - if too many bench games, baselines will be skewed low
            bench_ratio = (minutes < 15).sum() / len(minutes)
            if bench_ratio > 0.3:
                print(f"   🚨 HIGH BENCH GAME RATIO ({bench_ratio:.1%}) - Will skew baselines low!")
            else:
                print(f"   ✅ Reasonable bench/starter ratio")

        # 5. Specific Player Deep Dive
        print(f"\n🎯 DEEP DIVE: PROBLEMATIC PLAYERS")
        print("=" * 50)

        problematic_players = ['Nikola Jokic', 'James Harden', 'Kevin Durant']

        for player in problematic_players:
            print(f"\n🏀 {player}:")
            player_data = data[data['fullName'] == player]

            if len(player_data) == 0:
                print(f"   ❌ NO DATA FOUND")
                continue

            print(f"   Total Games: {len(player_data):,}")

            # Recent games analysis
            if 'gameDate' in data.columns:
                recent_cutoff = pd.Timestamp.now().tz_localize('UTC') - pd.Timedelta(days=30)
                player_dates = pd.to_datetime(player_data['gameDate'], errors='coerce')
                recent_games = player_data[player_dates > recent_cutoff]
                print(f"   Recent Games (30d): {len(recent_games)}")

            # Minutes distribution
            if 'numMinutes' in player_data.columns:
                avg_minutes = player_data['numMinutes'].mean()
                starter_games = (player_data['numMinutes'] >= 28).sum()
                print(f"   Avg Minutes: {avg_minutes:.1f}")
                print(f"   Starter-level Games: {starter_games} ({starter_games/len(player_data)*100:.1f}%)")

            # Stat averages (all games vs starter games)
            for stat in ['points', 'reboundsTotal', 'assists']:
                if stat in player_data.columns:
                    all_avg = player_data[stat].mean()
                    starter_avg = player_data[player_data['numMinutes'] >= 28][stat].mean()
                    print(f"   {stat:12s}: All={all_avg:5.1f}, Starter={starter_avg:5.1f} (Δ={starter_avg-all_avg:+5.1f})")

            # Identify data quality issues
            issues = []
            if avg_minutes < 25:
                issues.append("Low average minutes")
            if starter_games / len(player_data) < 0.6:
                issues.append("Too many bench games")
            if len(recent_games) < 10:
                issues.append("Insufficient recent data")

            if issues:
                print(f"   🚨 ISSUES: {', '.join(issues)}")
            else:
                print(f"   ✅ Data quality good")

        # 6. Feature Engineering Data Issues
        print(f"\n⚙️  FEATURE ENGINEERING AUDIT")
        print("=" * 50)

        # Check for columns that might cause NaN in calculations
        potential_nan_columns = []
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                nan_count = data[col].isnull().sum()
                inf_count = np.isinf(data[col]).sum()
                if nan_count > 0 or inf_count > 0:
                    potential_nan_columns.append((col, nan_count, inf_count))

        if potential_nan_columns:
            print("   🚨 COLUMNS WITH NaN/INF VALUES (will break predictions):")
            for col, nan_cnt, inf_cnt in potential_nan_columns[:10]:  # Show top 10
                print(f"      {col:20s}: {nan_cnt:6,} NaN, {inf_cnt:6,} Inf")
        else:
            print("   ✅ No NaN/Inf values detected in numeric columns")

        # 7. Team/Opponent Data Quality
        print(f"\n🏟️  TEAM DATA ANALYSIS:")
        if 'playerteamName' in data.columns:
            unique_teams = data['playerteamName'].nunique()
            print(f"   Unique Teams: {unique_teams}")

            # Check for team name consistency
            team_counts = data['playerteamName'].value_counts()
            print(f"   Most represented team: {team_counts.index[0]} ({team_counts.iloc[0]:,} games)")

            # Look for obvious team name issues
            team_names = set(data['playerteamName'].unique())
            suspicious_names = [name for name in team_names if len(str(name)) < 3 or pd.isna(name)]
            if suspicious_names:
                print(f"   🚨 Suspicious team names: {suspicious_names}")

        # 8. Raw Data Recommendations
        print(f"\n💡 RAW DATA UTILIZATION RECOMMENDATIONS")
        print("=" * 50)
        print("Based on audit findings:")

        recommendations = [
            "✓ Implement better date parsing and validation",
            "✓ Filter out games with <25 minutes to improve baselines",
            "✓ Add recency weighting (prefer last 30 days)",
            "✓ Handle NaN values before feature engineering",
            "✓ Validate team name consistency",
            "✓ Add data freshness checks and alerts",
            "✓ Implement per-minute projections for partial games",
            "✓ Add injury/rest day detection",
            "✓ Include usage rate and pace adjustments",
            "✓ Add position-based normalization"
        ]

        for rec in recommendations:
            print(f"   {rec}")

        print(f"\n🎯 PRIORITY FIXES TO IMPLEMENT:")
        print("   1. Fix NaN handling in feature engineering")
        print("   2. Improve date parsing and recency weighting")
        print("   3. Filter low-minute games more aggressively")
        print("   4. Add data freshness validation")
        print("   5. Implement robust error handling")

    except Exception as e:
        print(f"❌ Audit failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    audit_raw_data()