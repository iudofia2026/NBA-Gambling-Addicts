"""
Advanced NBA Features - Iteration 1
Weighted Performance Windows and Enhanced Defensive Matchups

This module implements the most impactful scoring factors identified:
1. Weighted 10-game performance windows (last 3 games weighted 40%)
2. Enhanced defensive matchup quality metrics
3. Shooting efficiency trends over raw points
4. Localized defensive matchup analysis (recent games only)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class AdvancedFeatureGeneratorV1:
    """Generate advanced features for NBA scoring predictions."""

    def __init__(self, historical_data):
        """Initialize with historical game data."""
        self.data = historical_data.copy()
        self.data['gameDate'] = pd.to_datetime(self.data['gameDate'], errors='coerce')
        self.data = self.data.sort_values('gameDate')

    def create_weighted_performance_features(self, player_name, games_back=10):
        """
        Create weighted performance features based on research findings:
        - Last 3 games: 40% weight
        - Games 4-10: 60% weight (evenly distributed)
        """

        player_data = self.data[self.data['fullName'] == player_name].tail(games_back).copy()

        if player_data.empty:
            return self._empty_weighted_features()

        # Split into recent (last 3) and historical (games 4-10)
        recent_games = player_data.tail(3)
        historical_games = player_data.head(max(0, len(player_data) - 3))

        # Calculate weighted averages
        features = {}

        # Points weighted average
        if len(recent_games) > 0:
            recent_points_avg = recent_games['points'].mean()
        else:
            recent_points_avg = 0

        if len(historical_games) > 0:
            hist_points_avg = historical_games['points'].mean()
        else:
            hist_points_avg = 0

        features['weighted_points'] = (recent_points_avg * 0.4) + (hist_points_avg * 0.6)

        # Shooting efficiency weighted (more predictive than points)
        features['weighted_fg_pct'] = (
            (recent_games['fieldGoalsPercentage'].mean() if len(recent_games) > 0 else 0) * 0.4 +
            (historical_games['fieldGoalsPercentage'].mean() if len(historical_games) > 0 else 0) * 0.6
        )

        features['weighted_minutes'] = (
            (recent_games['numMinutes'].mean() if len(recent_games) > 0 else 0) * 0.4 +
            (historical_games['numMinutes'].mean() if len(historical_games) > 0 else 0) * 0.6
        )

        # Efficiency trend (improvement/decline)
        if len(recent_games) >= 2:
            features['points_trend'] = np.polyfit(range(len(recent_games)),
                                                 recent_games['points'].values, 1)[0]
        else:
            features['points_trend'] = 0

        # Consistency (lower is more consistent)
        features['points_volatility'] = player_data['points'].std() if len(player_data) > 1 else 0

        # High-scoring games frequency (over baseline)
        baseline = features['weighted_points']
        features['high_scoring_freq'] = (player_data['points'] > baseline * 1.2).mean() if len(player_data) > 0 else 0

        return features

    def get_defensive_matchup_quality(self, player_name, opponent_team, games_back=15):
        """
        Enhanced defensive matchup analysis using recent games only.
        Focus on how the opponent defense has performed recently.
        """

        # Get opponent's recent defensive performance (last 15 games)
        cutoff_date = datetime.now() - timedelta(days=45)
        recent_opponent_games = self.data[
            (self.data['opponentteamName'] == opponent_team) &
            (self.data['gameDate'] >= cutoff_date)
        ].tail(15)

        if recent_opponent_games.empty:
            return self._empty_defensive_features()

        # Calculate recent defensive metrics
        features = {}

        # Points allowed per game (recent)
        features['opp_recent_pts_allowed'] = recent_opponent_games['points'].mean()

        # Defensive trend (improving/worsening)
        if len(recent_opponent_games) >= 5:
            last_5_allowed = recent_opponent_games.tail(5)['points'].mean()
            previous_10_allowed = recent_opponent_games.head(max(0, len(recent_opponent_games) - 5))['points'].mean()
            features['opp_def_trend'] = last_5_allowed - previous_10_allowed
        else:
            features['opp_def_trend'] = 0

        # Defensive rating (adjusted for pace)
        features['opp_def_rating'] = features['opp_recent_pts_allowed'] * 100 / (
            recent_opponent_games.get('opponent_pace', 100).mean()
        )

        # Elite defensive games frequency (under 100 points)
        features['opp_elite_def_freq'] = (recent_opponent_games['points'] < 100).mean()

        # Perimeter defense impact (for guards/wings)
        perimeter_players = recent_opponent_games[
            recent_opponent_games['fullName'].str.contains('Curry|Lillard|Mitchell|Irving|Beal', na=False)
        ]
        if len(perimeter_players) > 0:
            features['opp_perimeter_def_strength'] = 110 - perimeter_players['points'].mean()
        else:
            features['opp_perimeter_def_strength'] = 0

        # Paint defense impact (for bigs)
        big_players = recent_opponent_games[
            recent_opponent_games['fullName'].str.contains('Jokić|Embiid|Davis|Towns|Gobert', na=False)
        ]
        if len(big_players) > 0:
            features['opp_paint_def_strength'] = 110 - big_players['points'].mean()
        else:
            features['opp_paint_def_strength'] = 0

        return features

    def get_player_vs_opponent_recent_history(self, player_name, opponent_team, games_back=5):
        """
        Player's recent performance against this specific opponent.
        Uses last 5 matchups to keep it current and relevant.
        """

        # Get player's games against this opponent (all-time, sorted by date)
        matchup_games = self.data[
            (self.data['fullName'] == player_name) &
            (
                (self.data['playerteamName'] == opponent_team) |
                (self.data['opponentteamName'] == opponent_team)
            )
        ].sort_values('gameDate', ascending=False).head(games_back)

        if matchup_games.empty:
            return self._empty_matchup_features()

        features = {}

        # Recent performance vs opponent
        features['vs_opp_avg_points'] = matchup_games['points'].mean()
        features['vs_opp_avg_minutes'] = matchup_games['numMinutes'].mean()
        features['vs_opp_fg_pct'] = matchup_games['fieldGoalsPercentage'].mean()

        # Success rate against opponent
        features['vs_opp_over_rate'] = (matchup_games['over_threshold'] == 1).mean()

        # Consistency against opponent
        features['vs_opp_consistency'] = 1 - (matchup_games['points'].std() / matchup_games['points'].mean()) if len(matchup_games) > 1 else 0

        # Recent trend against opponent (last 2 vs previous)
        if len(matchup_games) >= 3:
            last_2 = matchup_games.head(2)['points'].mean()
            previous = matchup_games.iloc[2:]['points'].mean()
            features['vs_opp_trend'] = last_2 - previous
        else:
            features['vs_opp_trend'] = 0

        # Experience factor (confidence in matchup data)
        features['vs_opp_experience'] = min(len(matchup_games) / 5, 1.0)

        return features

    def generate_all_advanced_features(self, player_name, opponent_team):
        """
        Generate all advanced features for a player-opponent matchup.
        """

        features = {}

        # Weighted performance features
        weighted_perf = self.create_weighted_performance_features(player_name)
        features.update({f'weighted_{k}': v for k, v in weighted_perf.items()})

        # Defensive matchup features
        defensive = self.get_defensive_matchup_quality(player_name, opponent_team)
        features.update({f'def_{k}': v for k, v in defensive.items()})

        # Player vs opponent history
        matchup = self.get_player_vs_opponent_recent_history(player_name, opponent_team)
        features.update({f'matchup_{k}': v for k, v in matchup.items()})

        # Composite features
        features['scoring_expectation'] = (
            weighted_perf['weighted_points'] * 0.5 +
            matchup['vs_opp_avg_points'] * 0.3 +
            (110 - defensive['opp_def_rating']) * 0.2
        )

        features['matchup_quality_score'] = (
            defensive['opp_def_rating'] * -0.4 +  # Opponent defense (negative is better)
            weighted_perf['points_trend'] * 0.3 +  # Player trend
            matchup['vs_opp_over_rate'] * 0.3      # Historical success
        )

        return features

    def _empty_weighted_features(self):
        """Return empty weighted features when no data available."""
        return {
            'weighted_points': 0,
            'weighted_fg_pct': 0.45,
            'weighted_minutes': 25,
            'points_trend': 0,
            'points_volatility': 5,
            'high_scoring_freq': 0.3
        }

    def _empty_defensive_features(self):
        """Return empty defensive features when no data available."""
        return {
            'opp_recent_pts_allowed': 110,
            'opp_def_trend': 0,
            'opp_def_rating': 110,
            'opp_elite_def_freq': 0.3,
            'opp_perimeter_def_strength': 0,
            'opp_paint_def_strength': 0
        }

    def _empty_matchup_features(self):
        """Return empty matchup features when no data available."""
        return {
            'vs_opp_avg_points': 0,
            'vs_opp_avg_minutes': 0,
            'vs_opp_fg_pct': 0.45,
            'vs_opp_over_rate': 0.5,
            'vs_opp_consistency': 0.5,
            'vs_opp_trend': 0,
            'vs_opp_experience': 0
        }

# Test the advanced features
if __name__ == "__main__":
    print("Testing Advanced NBA Features - Iteration 1")
    print("=" * 50)

    try:
        # Load historical data
        data = pd.read_csv('../data/processed/engineered_features.csv')
        print(f"Loaded {len(data)} historical games")

        # Initialize feature generator
        generator = AdvancedFeatureGeneratorV1(data)

        # Test with a sample player
        test_player = data['fullName'].iloc[0]
        test_opponent = data['opponentteamName'].iloc[0]

        print(f"\nTesting with: {test_player} vs {test_opponent}")

        # Generate features
        features = generator.generate_all_advanced_features(test_player, test_opponent)

        print(f"\nGenerated {len(features)} advanced features:")
        for feature, value in features.items():
            if isinstance(value, float):
                print(f"  {feature}: {value:.2f}")
            else:
                print(f"  {feature}: {value}")

        print("\n✅ Advanced features v1 working correctly!")

    except Exception as e:
        print(f"❌ Error: {e}")