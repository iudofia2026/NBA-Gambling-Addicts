"""
Shot Quality and Team Dynamics - Iteration 2
Advanced metrics for shot quality, lineup spacing, and clutch performance

Based on 2025 research:
- Well-spaced lineups: +6.3 points per 100 possessions
- 4-out/5-out spacing: 12.8% efficiency increase
- Optimal spacing: 18+ feet between non-ball handlers
- Clutch performance varies significantly by player
"""

import pandas as pd
import numpy as np

class ShotQualityFeaturesV2:
    """Generate shot quality and team dynamics features."""

    def __init__(self, historical_data):
        """Initialize with historical game data."""
        self.data = historical_data.copy()
        self.data['gameDate'] = pd.to_datetime(self.data['gameDate'], errors='coerce')
        self.data = self.data.sort_values('gameDate')

    def calculate_shot_quality_metrics(self, player_name, games_back=15):
        """
        Calculate advanced shot quality metrics based on research findings.
        """

        player_data = self.data[self.data['fullName'] == player_name].tail(games_back)

        if player_data.empty:
            return self._empty_shot_quality_features()

        features = {}

        # True Shooting Percentage (TS%)
        player_data['ts_pct'] = (
            player_data['points'] / (
                2 * (player_data['fieldGoalsAttempted'] + 0.44 * player_data['freeThrowsAttempted'])
            )
        ).fillna(0.5)

        features['avg_ts_pct'] = player_data['ts_pct'].mean()
        features['recent_ts_trend'] = self._calculate_trend(player_data['ts_pct'])

        # Effective Field Goal Percentage (eFG%)
        player_data['efg_pct'] = (
            (player_data['fieldGoalsMade'] + 0.5 * player_data['threePointersMade']) /
            player_data['fieldGoalsAttempted']
        ).fillna(0.5)

        features['avg_efg_pct'] = player_data['efg_pct'].mean()
        features['recent_efg_trend'] = self._calculate_trend(player_data['efg_pct'])

        # Shot Distribution
        three_pa_rate = (player_data['threePointersAttempted'] / player_data['fieldGoalsAttempted']).mean()
        features['three_point_attempt_rate'] = three_pa_rate
        features['three_point_efficiency'] = (player_data['threePointersPercentage'].mean() * three_pa_rate)

        # Paint Efficiency
        paint_attempts = player_data['fieldGoalsAttempted'] - player_data['threePointersAttempted']
        paint_made = player_data['fieldGoalsMade'] - player_data['threePointersMade']
        features['paint_fg_pct'] = (paint_made / paint_attempts).mean() if paint_attempts.sum() > 0 else 0.5

        # Free Throw Rate (indicator of aggressive play/driving)
        features['ft_rate'] = (player_data['freeThrowsAttempted'] / player_data['fieldGoalsAttempted']).mean()
        features['ft_pct'] = player_data['freeThrowsPercentage'].mean()

        # Shot Selection Quality Score
        features['shot_quality_score'] = (
            features['avg_ts_pct'] * 0.4 +
            features['avg_efg_pct'] * 0.3 +
            np.minimum(features['three_point_efficiency'] * 100, 1.0) * 0.3
        )

        return features

    def calculate_clutch_performance(self, player_name, games_back=30):
        """
        Analyze clutch performance (last 5 minutes, close games).
        Based on research: clutch performance varies significantly by player.
        """

        # Since we don't have play-by-play data, simulate clutch indicators
        player_data = self.data[self.data['fullName'] == player_name].tail(games_back)

        if player_data.empty:
            return self._empty_clutch_features()

        features = {}

        # Clutch indicators from recent high-pressure games
        # Simulate: high minutes + close margin games
        high_minutes_games = player_data[player_data['numMinutes'] >= 35]

        if len(high_minutes_games) > 0:
            features['clutch_points'] = high_minutes_games['points'].mean()
            features['clutch_efficiency'] = (high_minutes_games['points'] / high_minutes_games['numMinutes']).mean()
            features['clutch_fg_pct'] = high_minutes_games['fieldGoalsPercentage'].mean()
        else:
            # Use overall performance as proxy
            features['clutch_points'] = player_data['points'].mean()
            features['clutch_efficiency'] = (player_data['points'] / player_data['numMinutes']).mean()
            features['clutch_fg_pct'] = player_data['fieldGoalsPercentage'].mean()

        # Clutch trend (improvement in pressure situations)
        if len(high_minutes_games) >= 5:
            recent_clutch = high_minutes_games.tail(3)['points'].mean()
            earlier_clutch = high_minutes_games.head(len(high_minutes_games) - 3)['points'].mean()
            features['clutch_trend'] = recent_clutch - earlier_clutch
        else:
            features['clutch_trend'] = 0

        # Late game performance (4th quarter focus)
        # Simulate with overall 4th quarter performance
        features['fourth_quarter_rate'] = player_data['numMinutes'].mean() / 48  # Estimation
        features['closing_ability'] = features['clutch_efficiency'] * features['fourth_quarter_rate']

        # Clutch reliability (consistency in pressure)
        features['clutch_consistency'] = 1 - (high_minutes_games['points'].std() / high_minutes_games['points'].mean()) if len(high_minutes_games) > 1 else 0.5

        return features

    def calculate_lineup_spacing_impact(self, player_name, player_team, games_back=20):
        """
        Estimate lineup spacing impact based on research:
        - 4-out/5-out spacing: 12.8% efficiency increase
        - Optimal spacing: 18+ feet between non-ball handlers
        """

        # Since we don't have lineup data, simulate based on team composition
        team_data = self.data[
            (self.data['playerteamName'] == player_team) |
            (self.data['opponentteamName'] == player_team)
        ].tail(games_back)

        if team_data.empty:
            return self._empty_spacing_features()

        features = {}

        # Estimate team spacing based on shooting personnel
        # Count shooters (players with >35% 3PT% and significant attempts)
        team_shooters = team_data[team_data['threePointersPercentage'] > 0.35]
        team_shooters = team_shooters[team_shooters['threePointersAttempted'] / team_shooters['fieldGoalsAttempted'] > 0.3]

        num_shooters = len(team_shooters['fullName'].unique())
        features['team_shooter_count'] = num_shooters

        # Spacing configuration impact
        if num_shooters >= 4:
            features['spacing_config'] = 'five_out'  # 5-out spacing
            features['spacing_efficiency_bonus'] = 0.128  # 12.8% increase
        elif num_shooters == 3:
            features['spacing_config'] = 'four_out'  # 4-out spacing
            features['spacing_efficiency_bonus'] = 0.08  # 8% increase (estimated)
        else:
            features['spacing_config'] = 'traditional'
            features['spacing_efficiency_bonus'] = 0

        # Player specific spacing benefit
        player_shooting = team_data[team_data['fullName'] == player_name]
        if len(player_shooting) > 0:
            player_is_shooter = (
                (player_shooting['threePointersPercentage'].mean() > 0.35) &
                (player_shooting['threePointersAttempted'].mean() / player_shooting['fieldGoalsAttempted'].mean() > 0.3)
            )
            features['player_spacing_benefit'] = features['spacing_efficiency_bonus'] if player_is_shooter else features['spacing_efficiency_bonus'] * 0.6
        else:
            features['player_spacing_benefit'] = features['spacing_efficiency_bonus']

        # Driving opportunity factor (based on research: 23% more driving opportunities)
        features['driving_opportunities'] = 1.0 + (features['spacing_efficiency_bonus'] * 0.5)

        # Close-range shot attempts increase (34% in well-spaced lineups)
        features['paint_attempts_factor'] = 1.0 + (features['spacing_efficiency_bonus'] * 0.8)

        return features

    def calculate_usage_efficiency(self, player_name, games_back=15):
        """
        Calculate usage rate and efficiency based on research:
        High usage with high efficiency = star player
        High usage with low efficiency = inefficient scorer
        """

        player_data = self.data[self.data['fullName'] == player_name].tail(games_back)

        if player_data.empty:
            return self._empty_usage_features()

        features = {}

        # Estimate usage rate (simplified)
        # Usage ≈ (FGA + 0.44 * FTA + TOV) / Team Possessions
        # We'll use a proxy based on shot attempts and minutes
        team_games = self.data.tail(games_back * 10)  # Rough team sample

        if len(team_games) > 0:
            team_fga_per_game = team_games['fieldGoalsAttempted'].mean() * 5  # 5 players per team
            player_fga = player_data['fieldGoalsAttempted'].mean()
            features['usage_rate'] = min(player_fga / team_fga_per_game * 5, 0.4)  # Cap at 40%
        else:
            features['usage_rate'] = 0.2  # League average

        # Usage efficiency (points per usage)
        features['usage_efficiency'] = player_data['points'].mean() / (features['usage_rate'] * 100)

        # Usage categories based on efficiency
        if features['usage_rate'] > 0.25:
            if features['usage_efficiency'] > 1.2:
                features['usage_type'] = 'high_efficiency_star'
            elif features['usage_efficiency'] > 1.0:
                features['usage_type'] = 'efficient_primary'
            else:
                features['usage_type'] = 'volume_scorer'
        elif features['usage_rate'] > 0.15:
            features['usage_type'] = 'secondary_scorer'
        else:
            features['usage_type'] = 'role_player'

        # Scoring role multiplier for predictions
        usage_multipliers = {
            'high_efficiency_star': 1.25,
            'efficient_primary': 1.15,
            'volume_scorer': 1.05,
            'secondary_scorer': 1.0,
            'role_player': 0.95
        }
        features['usage_role_multiplier'] = usage_multipliers.get(features['usage_type'], 1.0)

        # Fatigue factor based on usage
        features['usage_fatigue_factor'] = 1.0 + (features['usage_rate'] * 0.3)

        return features

    def generate_all_shot_quality_features(self, player_name, player_team):
        """Generate all shot quality and team dynamics features."""

        features = {}

        # Shot quality metrics
        shot_quality = self.calculate_shot_quality_metrics(player_name)
        features.update({f'shot_{k}': v for k, v in shot_quality.items()})

        # Clutch performance
        clutch = self.calculate_clutch_performance(player_name)
        features.update({f'clutch_{k}': v for k, v in clutch.items()})

        # Lineup spacing impact
        spacing = self.calculate_lineup_spacing_impact(player_name, player_team)
        features.update({f'spacing_{k}': v for k, v in spacing.items()})

        # Usage efficiency
        usage = self.calculate_usage_efficiency(player_name)
        features.update({f'usage_{k}': v for k, v in usage.items()})

        # Composite features
        features['adjusted_scoring_potential'] = (
            shot_quality['shot_quality_score'] * usage['usage_role_multiplier'] *
            (1 + spacing['player_spacing_benefit']) *
            (1 + clutch['clutch_efficiency'] * 0.3)
        )

        features['game_situation_multiplier'] = (
            (1 + spacing['driving_opportunities'] * 0.2) +  # More drives = more scoring chances
            (1 + clutch['closing_ability'] * 0.1) +      # Clutch factor
            (1 - usage['usage_fatigue_factor'] * 0.05)     # Fatigue penalty
        )

        return features

    # Helper methods
    def _calculate_trend(self, series):
        """Calculate trend for a series (positive = improving)."""
        if len(series) < 2:
            return 0
        return np.polyfit(range(len(series)), series, 1)[0]

    def _empty_shot_quality_features(self):
        return {
            'avg_ts_pct': 0.55,
            'recent_ts_trend': 0,
            'avg_efg_pct': 0.52,
            'recent_efg_trend': 0,
            'three_point_attempt_rate': 0.35,
            'three_point_efficiency': 0.12,
            'paint_fg_pct': 0.55,
            'ft_rate': 0.25,
            'ft_pct': 0.75,
            'shot_quality_score': 0.5
        }

    def _empty_clutch_features(self):
        return {
            'clutch_points': 0,
            'clutch_efficiency': 0.3,
            'clutch_fg_pct': 0.45,
            'clutch_trend': 0,
            'fourth_quarter_rate': 0.25,
            'closing_ability': 0.075,
            'clutch_consistency': 0.5
        }

    def _empty_spacing_features(self):
        return {
            'team_shooter_count': 2,
            'spacing_config': 'traditional',
            'spacing_efficiency_bonus': 0,
            'player_spacing_benefit': 0,
            'driving_opportunities': 1.0,
            'paint_attempts_factor': 1.0
        }

    def _empty_usage_features(self):
        return {
            'usage_rate': 0.2,
            'usage_efficiency': 1.0,
            'usage_type': 'secondary_scorer',
            'usage_role_multiplier': 1.0,
            'usage_fatigue_factor': 1.06
        }

# Test the shot quality features
if __name__ == "__main__":
    print("Testing Shot Quality Features - Iteration 2")
    print("=" * 45)

    try:
        data = pd.read_csv('../data/processed/engineered_features.csv')
        generator = ShotQualityFeaturesV2(data)

        test_player = data['fullName'].iloc[0]
        test_team = data['playerteamName'].iloc[0]

        print(f"\nTesting with: {test_player} ({test_team})")

        features = generator.generate_all_shot_quality_features(test_player, test_team)

        print(f"\nGenerated {len(features)} shot quality features:")
        for feature, value in features.items():
            if isinstance(value, float):
                print(f"  {feature}: {value:.3f}")
            else:
                print(f"  {feature}: {value}")

        print("\n✅ Shot quality v2 features working!")

    except Exception as e:
        print(f"Error: {e}")