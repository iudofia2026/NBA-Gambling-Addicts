"""
Matchup Analytics - Iteration 5
Individual Player Home/Away Effects & Statistically Significant Player-Player Matchups

Based on research:
- Individual players show varying H/A effects (some significant, p<0.05)
- Player-player matchups need 10+ games for significance
- 30+ games ideal for reliable trends
- Sample size validation critical
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MatchupAnalyticsV5:
    """Advanced matchup analytics with statistical validation."""

    def __init__(self, historical_data):
        """Initialize with historical data."""
        self.data = historical_data.copy()
        self.data['gameDate'] = pd.to_datetime(self.data['gameDate'], errors='coerce')
        self.data = self.data.sort_values('gameDate')

        # Minimum sample sizes based on research
        self.MIN_GAMES_FOR_MATCHUP = 10
        self.MIN_GAMES_FOR_TREND = 30
        self.MIN_GAMES_FOR_RELIABLE = 50

    def calculate_individual_home_away_effects(self, player_name):
        """
        Calculate individual player's home/away performance with statistical significance.
        Research shows players vary significantly in H/A effects.
        """
        player_data = self.data[self.data['fullName'] == player_name].copy()

        if len(player_data) < 20:  # Minimum for any meaningful analysis
            return self._get_default_ha_effects()

        home_games = player_data[player_data['home'] == 1]
        away_games = player_data[player_data['home'] == 0]

        if len(home_games) < 5 or len(away_games) < 5:
            return self._get_default_ha_effects()

        features = {}

        # Calculate key metrics
        metrics = ['points', 'numMinutes', 'fieldGoalsPercentage', 'threePointersPercentage',
                   'freeThrowsPercentage', 'reboundsTotal', 'assists', 'turnovers']

        for metric in metrics:
            if metric in home_games.columns and metric in away_games.columns:
                home_avg = home_games[metric].mean()
                away_avg = away_games[metric].mean()

                features[f'ha_{metric}_diff'] = home_avg - away_avg
                features[f'ha_{metric}_home_avg'] = home_avg
                features[f'ha_{metric}_away_avg'] = away_avg

                # Statistical significance test
                if len(home_games) >= 10 and len(away_games) >= 10:
                    try:
                        t_stat, p_value = stats.ttest_ind(
                            home_games[metric].dropna(),
                            away_games[metric].dropna()
                        )
                        features[f'ha_{metric}_significant'] = p_value < 0.05
                        features[f'ha_{metric}_p_value'] = p_value
                        features[f'ha_{metric}_t_stat'] = t_stat
                    except:
                        features[f'ha_{metric}_significant'] = False
                        features[f'ha_{metric}_p_value'] = 1.0
                        features[f'ha_{metric}_t_stat'] = 0
                else:
                    features[f'ha_{metric}_significant'] = False
                    features[f'ha_{metric}_p_value'] = 1.0
                    features[f'ha_{metric}_t_stat'] = 0

        # Calculate Home/Away Index (0 = no difference, 1 = strong home advantage)
        significant_metrics = [m for m in metrics if features.get(f'ha_{m}_significant', False)]

        if significant_metrics:
            avg_significance = np.mean([features[f'ha_{m}_t_stat'] for m in significant_metrics])
            features['ha_home_advantage_index'] = abs(avg_significance) / 2.0
        else:
            features['ha_home_advantage_index'] = 0.0

        # Sample size confidence
        total_games = len(home_games) + len(away_games)
        features['ha_sample_confidence'] = min(total_games / self.MIN_GAMES_FOR_RELIABLE, 1.0)

        # Overall home advantage (weighted by significance)
        weighted_ha = 0
        total_weight = 0
        for metric in ['points', 'fieldGoalsPercentage', 'reboundsTotal', 'assists']:
            if features.get(f'ha_{metric}_significant', False):
                weighted_ha += features.get(f'ha_{metric}_diff', 0)
                total_weight += 1

        features['ha_weighted_advantage'] = weighted_ha / total_weight if total_weight > 0 else 0

        return features

    def analyze_player_player_matchup(self, player1_name, player2_name):
        """
        Analyze player vs player matchup with statistical significance testing.
        Only consider matchups with sufficient sample size based on research.
        """
        # Find games where both players played
        player1_games = self.data[self.data['fullName'] == player1_name]
        player2_games = self.data[self.data['fullName'] == player2_name]

        if len(player1_games) < 10 or len(player2_games) < 10:
            return self._get_default_matchup()

        # Find direct matchups (opponents)
        direct_matchups = []
        for _, game in player1_games.iterrows():
            opponent_players = self.data[
                (self.data['gameDate'] == game['gameDate']) &
                (self.data['opponentteamName'] == game['playerteamName'])
            ]
            if player2_name in opponent_players['fullName'].values:
                direct_matchups.append(game)

        if len(direct_matchups) < self.MIN_GAMES_FOR_MATCHUP:
            return self._get_default_matchup()

        # Get player2's stats in those games
        p1_matchup_stats = []
        p2_matchup_stats = []

        for matchup in direct_matchups:
            p1_stats = matchup.to_dict()
            p2_in_game = self.data[
                (self.data['gameDate'] == matchup['gameDate']) &
                (self.data['fullName'] == player2_name) &
                (self.data['opponentteamName'] == matchup['playerteamName'])
            ]

            if len(p2_in_game) > 0:
                p2_stats = p2_in_game.iloc[0].to_dict()
                p1_matchup_stats.append(p1_stats)
                p2_matchup_stats.append(p2_stats)

        if len(p1_matchup_stats) < self.MIN_GAMES_FOR_MATCHUP:
            return self._get_default_matchup()

        features = {}

        # Convert to DataFrames
        p1_df = pd.DataFrame(p1_matchup_stats)
        p2_df = pd.DataFrame(p2_matchup_stats)

        # Analyze matchup metrics
        matchup_metrics = ['points', 'numMinutes', 'fieldGoalsPercentage', 'threePointersPercentage']

        for metric in matchup_metrics:
            if metric in p1_df.columns and metric in p2_df.columns:
                p1_avg = p1_df[metric].mean()
                p2_avg = p2_df[metric].mean()

                features[f'match_{metric}_p1_avg'] = p1_avg
                features[f'match_{metric}_p2_avg'] = p2_avg
                features[f'match_{metric}_diff'] = p1_avg - p2_avg

                # Statistical significance
                if len(p1_df) >= 10 and len(p2_df) >= 10:
                    try:
                        t_stat, p_value = stats.ttest_ind(
                            p1_df[metric].dropna(),
                            p2_df[metric].dropna()
                        )
                        features[f'match_{metric}_significant'] = p_value < 0.05
                        features[f'match_{metric}_p_value'] = p_value
                        features[f'match_{metric}_t_stat'] = t_stat
                    except:
                        features[f'match_{metric}_significant'] = False
                        features[f'match_{metric}_p_value'] = 1.0
                        features[f'match_{metric}_t_stat'] = 0

        # Matchup dominance score
        significant_points = features.get('match_points_significant', False)
        if significant_points:
            features['match_dominance_score'] = features.get('match_points_t_stat', 0) / 2.0
        else:
            features['match_dominance_score'] = 0

        # Sample size confidence
        features['match_sample_size'] = len(direct_matchups)
        features['match_confidence'] = min(len(direct_matchups) / self.MIN_GAMES_FOR_RELIABLE, 1.0)

        # Historical record
        p1_wins = sum(1 for m in p1_matchup_stats if m.get('win', 0) == 1)
        features['match_p1_win_rate'] = p1_wins / len(p1_matchup_stats)

        # Consistency scoring
        p1_std = p1_df['points'].std() if len(p1_df) > 1 else 0
        p2_std = p2_df['points'].std() if len(p2_df) > 1 else 0
        p1_avg = p1_df['points'].mean()

        features['match_p1_consistency'] = 1 - (p1_std / p1_avg) if p1_avg > 0 else 0
        features['match_p2_consistency'] = 1 - (p2_std / p2_avg) if p2_avg > 0 else 0

        return features

    def get_opponent_specific_history(self, player_name, opponent_team, days_back=730):
        """
        Get player's performance against specific opponent with sample size validation.
        Recent history (last 2 years) only.
        """
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_back)

        opponent_games = self.data[
            (self.data['fullName'] == player_name) &
            ((self.data['opponentteamName'] == opponent_team) |
             (self.data['playerteamName'] == opponent_team)) &
            (self.data['gameDate'] >= cutoff_date)
        ].copy()

        if len(opponent_games) < self.MIN_GAMES_FOR_MATCHUP:
            return self._get_default_opponent_history()

        features = {}

        # Performance metrics
        features['opp_points_avg'] = opponent_games['points'].mean()
        features['opp_points_std'] = opponent_games['points'].std()
        features['opp_minutes_avg'] = opponent_games['numMinutes'].mean()
        features['opp_fg_pct'] = opponent_games['fieldGoalsPercentage'].mean()
        features['opp_ft_pct'] = opponent_games['freeThrowsPercentage'].mean()

        # Over rate against opponent
        features['opp_over_rate'] = (opponent_games['over_threshold'] == 1).mean()

        # Recent trend (last 5 games)
        recent_games = opponent_games.tail(5)
        if len(recent_games) >= 3:
            recent_avg = recent_games['points'].mean()
            older_games = opponent_games.head(len(opponent_games) - 5)
            if len(older_games) > 0:
                older_avg = older_games['points'].mean()
                features['opp_trend'] = recent_avg - older_avg
            else:
                features['opp_trend'] = 0
        else:
            features['opp_trend'] = 0

        # Consistency (lower = more consistent)
        features['opp_consistency'] = 1 - (features['opp_points_std'] / features['opp_points_avg']) if features['opp_points_avg'] > 0 else 0

        # Sample size metrics
        features['opp_total_games'] = len(opponent_games)
        features['opp_sample_adequate'] = len(opponent_games) >= self.MIN_GAMES_FOR_TREND
        features['opp_sample_confidence'] = min(len(opponent_games) / self.MIN_GAMES_FOR_RELIABLE, 1.0)

        return features

    def generate_all_matchup_features(self, player_name, opponent_team, opposing_player=None):
        """Generate all matchup analytics features."""

        features = {}

        # Individual home/away effects
        ha_effects = self.calculate_individual_home_away_effects(player_name)
        features.update({f'ind_ha_{k}': v for k, v in ha_effects.items()})

        # Player-player matchup (if provided)
        if opposing_player:
            player_matchup = self.analyze_player_player_matchup(player_name, opposing_player)
            features.update({f'pp_{k}': v for k, v in player_matchup.items()})

        # Opponent team history
        opp_history = self.get_opponent_specific_history(player_name, opponent_team)
        features.update({f'opp_{k}': v for k, v in opp_history.items()})

        # Composite matchup score
        ha_adjustment = ha_effects.get('ha_weighted_advantage', 0) * ha_effects.get('ha_sample_confidence', 0.5)
        opp_adjustment = (opp_history.get('opp_points_avg', 0) - ha_effects.get('ha_points_avg', 0)) * opp_history.get('opp_sample_confidence', 0.5)

        features['composite_matchup_score'] = ha_adjustment + opp_adjustment

        # Matchup reliability score
        ha_reliable = ha_effects.get('ha_sample_confidence', 0.5) > 0.6
        opp_reliable = opp_history.get('opp_sample_adequate', False)

        features['matchup_reliable'] = 1.0 if ha_reliable and opp_reliable else 0.5

        return features

    def _get_default_ha_effects(self):
        return {
            'ha_points_diff': 0,
            'ha_points_significant': False,
            'ha_weighted_advantage': 0,
            'ha_home_advantage_index': 0,
            'ha_sample_confidence': 0.2
        }

    def _get_default_matchup(self):
        return {
            'match_points_diff': 0,
            'match_points_significant': False,
            'match_dominance_score': 0,
            'match_sample_size': 0,
            'match_confidence': 0.1,
            'match_p1_win_rate': 0.5,
            'match_p1_consistency': 0.5,
            'match_p2_consistency': 0.5
        }

    def _get_default_opponent_history(self):
        return {
            'opp_points_avg': 0,
            'opp_points_std': 10,
            'opp_minutes_avg': 25,
            'opp_fg_pct': 0.45,
            'opp_ft_pct': 0.75,
            'opp_over_rate': 0.5,
            'opp_trend': 0,
            'opp_consistency': 0.5,
            'opp_total_games': 0,
            'opp_sample_adequate': False,
            'opp_sample_confidence': 0.1
        }

# Test the matchup analytics
if __name__ == "__main__":
    print("Testing Matchup Analytics - Iteration 5")
    print("=" * 45)

    try:
        data = pd.read_csv('../data/processed/engineered_features.csv')
        analyzer = MatchupAnalyticsV5(data)

        # Test individual HA effects
        test_player = data['fullName'].iloc[0]
        ha_features = analyzer.calculate_individual_home_away_effects(test_player)
        print(f"\nIndividual H/A Effects for {test_player}:")
        print(f"  Points difference: {ha_features.get('ha_points_diff', 0):+.1f}")
        print(f"  Significant difference: {ha_features.get('ha_points_significant', False)}")
        print(f"  Sample confidence: {ha_features.get('ha_sample_confidence', 0):.2f}")

        # Test opponent history
        test_team = data['opponentteamName'].iloc[0]
        opp_features = analyzer.get_opponent_specific_history(test_player, test_team)
        print(f"\nOpponent History vs {test_team}:")
        print(f"  Average points: {opp_features.get('opp_points_avg', 0):.1f}")
        print(f"  Over rate: {opp_features.get('opp_over_rate', 0):.1%}")
        print(f"  Sample adequate: {opp_features.get('opp_sample_adequate', False)}")

        print("\n✅ Matchup analytics v5 working!")

    except Exception as e:
        print(f"Error: {e}")