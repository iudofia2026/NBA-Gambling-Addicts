"""
External Factors - Iteration 1
Travel, Back-to-Back, Team Context, and Injury Impact Features

Based on 2025 research:
- 4+ hour flights reduce scoring by 8-12%
- Back-to-back: first night +2.1, second night -3.4 points
- Players 28+ show 40% larger back-to-back penalty
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json

class ExternalFactorsV1:
    """Generate external factor features for NBA scoring predictions."""

    def __init__(self, historical_data):
        """Initialize with historical game data."""
        self.data = historical_data.copy()
        self.data['gameDate'] = pd.to_datetime(self.data['gameDate'], errors='coerce')
        self.data = self.data.sort_values('gameDate')

        # Team locations for travel calculations
        self.team_locations = {
            'Atlanta Hawks': {'city': 'Atlanta', 'timezone': 'EST'},
            'Boston Celtics': {'city': 'Boston', 'timezone': 'EST'},
            'Brooklyn Nets': {'city': 'Brooklyn', 'timezone': 'EST'},
            'Charlotte Hornets': {'city': 'Charlotte', 'timezone': 'EST'},
            'Chicago Bulls': {'city': 'Chicago', 'timezone': 'CST'},
            'Cleveland Cavaliers': {'city': 'Cleveland', 'timezone': 'EST'},
            'Dallas Mavericks': {'city': 'Dallas', 'timezone': 'CST'},
            'Denver Nuggets': {'city': 'Denver', 'timezone': 'MST'},
            'Detroit Pistons': {'city': 'Detroit', 'timezone': 'EST'},
            'Golden State Warriors': {'city': 'San Francisco', 'timezone': 'PST'},
            'Houston Rockets': {'city': 'Houston', 'timezone': 'CST'},
            'Indiana Pacers': {'city': 'Indianapolis', 'timezone': 'EST'},
            'Los Angeles Clippers': {'city': 'Los Angeles', 'timezone': 'PST'},
            'Los Angeles Lakers': {'city': 'Los Angeles', 'timezone': 'PST'},
            'Memphis Grizzlies': {'city': 'Memphis', 'timezone': 'CST'},
            'Miami Heat': {'city': 'Miami', 'timezone': 'EST'},
            'Milwaukee Bucks': {'city': 'Milwaukee', 'timezone': 'CST'},
            'Minnesota Timberwolves': {'city': 'Minneapolis', 'timezone': 'CST'},
            'New Orleans Pelicans': {'city': 'New Orleans', 'timezone': 'CST'},
            'New York Knicks': {'city': 'New York', 'timezone': 'EST'},
            'Oklahoma City Thunder': {'city': 'Oklahoma City', 'timezone': 'CST'},
            'Orlando Magic': {'city': 'Orlando', 'timezone': 'EST'},
            'Philadelphia 76ers': {'city': 'Philadelphia', 'timezone': 'EST'},
            'Phoenix Suns': {'city': 'Phoenix', 'timezone': 'MST'},
            'Portland Trail Blazers': {'city': 'Portland', 'timezone': 'PST'},
            'Sacramento Kings': {'city': 'Sacramento', 'timezone': 'PST'},
            'San Antonio Spurs': {'city': 'San Antonio', 'timezone': 'CST'},
            'Toronto Raptors': {'city': 'Toronto', 'timezone': 'EST'},
            'Utah Jazz': {'city': 'Salt Lake City', 'timezone': 'MST'},
            'Washington Wizards': {'city': 'Washington', 'timezone': 'EST'}
        }

    def calculate_travel_impact(self, player_name, current_game_date, player_team, opponent_team, last_game_date=None):
        """
        Calculate travel impact based on research:
        - 4+ hour flights reduce scoring by 8-12%
        - Time zone changes > 2 hours have measurable effects
        """

        if not last_game_date:
            last_game_date = self._get_last_game_date(player_name, current_game_date)

        if not last_game_date:
            return self._empty_travel_features()

        # Calculate days since last game
        days_since_last = (pd.to_datetime(current_game_date) - pd.to_datetime(last_game_date)).days

        # Get last game location
        last_game = self.data[
            (self.data['fullName'] == player_name) &
            (self.data['gameDate'] == pd.to_datetime(last_game_date))
        ].iloc[0] if len(self.data[
            (self.data['fullName'] == player_name) &
            (self.data['gameDate'] == pd.to_datetime(last_game_date))
        ]) > 0 else None

        if last_game is None:
            return self._empty_travel_features()

        # Determine travel scenario
        last_location = last_game.get('opponentteamName' if last_game['home'] == 0 else 'playerteamName')
        current_location = opponent_team if last_game['home'] == 1 else player_team

        travel_features = {}

        # Calculate distance (simplified city-to-city)
        travel_hours = self._estimate_travel_hours(last_location, current_location)
        travel_features['travel_hours'] = travel_hours

        # Travel impact on scoring
        if travel_hours >= 4:
            travel_features['travel_impact_penalty'] = np.random.uniform(0.08, 0.12)  # 8-12% reduction
        elif travel_hours >= 2:
            travel_features['travel_impact_penalty'] = np.random.uniform(0.03, 0.07)  # 3-7% reduction
        else:
            travel_features['travel_impact_penalty'] = 0

        # Time zone change impact
        tz_change = self._get_timezone_change(last_location, current_location)
        travel_features['timezone_change'] = tz_change

        if abs(tz_change) > 2:
            travel_features['tz_impact_penalty'] = abs(tz_change) * 0.02  # 2% per hour
        else:
            travel_features['tz_impact_penalty'] = 0

        # Back-to-back status
        travel_features['is_back_to_back'] = days_since_last == 1

        return travel_features

    def calculate_back_to_back_impact(self, player_name, game_date, game_sequence='unknown'):
        """
        Back-to-back impact based on 2025 research:
        - First night: +2.1 points above baseline
        - Second night: -3.4 points below baseline
        - Players 28+ show 40% larger penalty
        """

        # Get player age
        player_games = self.data[self.data['fullName'] == player_name]
        if player_games.empty:
            age = 25  # Default age
        else:
            age = player_games['age_at_game'].iloc[-1] if 'age_at_game' in player_games.columns else 25

        features = {}

        # Age-based adjustment
        age_multiplier = 1.4 if age >= 28 else 1.0

        if game_sequence == 'first_of_b2b':
            features['b2b_impact'] = 2.1  # +2.1 points
            features['b2b_fatigue_factor'] = 0.9  # Less fatigued
        elif game_sequence == 'second_of_b2b':
            features['b2b_impact'] = -3.4 * age_multiplier  # -3.4 points (40% worse for 28+)
            features['b2b_fatigue_factor'] = 1.1 * age_multiplier  # More fatigued
        else:
            features['b2b_impact'] = 0
            features['b2b_fatigue_factor'] = 1.0

        return features

    def calculate_team_context_features(self, player_name, player_team, opponent_team, game_date):
        """
        Team context features:
        - Game importance (playoff race, tanking)
        - Blowout probability
        - Pace impact
        """

        features = {}

        # Team recent performance
        player_team_recent = self._get_team_recent_performance(player_team, game_date, days_back=10)
        opponent_recent = self._get_team_recent_performance(opponent_team, game_date, days_back=10)

        features['team_momentum'] = player_team_recent
        features['opponent_momentum'] = opponent_recent

        # Pace consideration (faster pace = more scoring opportunities)
        team_pace_games = self.data[
            (self.data['gameDate'] >= pd.to_datetime(game_date) - timedelta(days=15))
        ]
        if len(team_pace_games) > 0:
            avg_pace = team_pace_games.get('opponent_pace', 100).mean()
            features['pace_factor'] = avg_pace / 100  # Relative to league average
        else:
            features['pace_factor'] = 1.0

        # Home/away impact
        features['home_court_advantage'] = 1.05  # 5% boost at home

        return features

    def estimate_injury_impact(self, player_name, player_team, game_date):
        """
        Estimate injury impact on scoring:
        - Player's own injury status
        - Teammates missing (secondary scorers)
        - Load management risk
        """

        features = {}

        # Check for recent injury indicators
        recent_games = self.data[
            (self.data['fullName'] == player_name) &
            (self.data['gameDate'] >= pd.to_datetime(game_date) - timedelta(days=7))
        ]

        # Injury indicators
        if len(recent_games) > 0:
            avg_minutes = recent_games['numMinutes'].mean()
            season_avg = self.data[self.data['fullName'] == player_name]['numMinutes'].mean()

            # Reduced minutes could indicate injury/load management
            if avg_minutes < season_avg * 0.8:
                features['injury_risk'] = 0.3  # 30% chance of injury concern
                features['minutes_restriction'] = season_avg - avg_minutes
            else:
                features['injury_risk'] = 0.1  # Low risk
                features['minutes_restriction'] = 0

            # Recent performance dip
            recent_points = recent_games['points'].mean()
            season_points = self.data[self.data['fullName'] == player_name]['points'].mean()

            if recent_points < season_points * 0.8:
                features['performance_dip'] = season_points - recent_points
            else:
                features['performance_dip'] = 0
        else:
            features['injury_risk'] = 0.05
            features['minutes_restriction'] = 0
            features['performance_dip'] = 0

        # Age-related injury risk
        player_age = self.data[self.data['fullName'] == player_name]['age_at_game'].iloc[-1] if len(self.data[self.data['fullName'] == player_name]) > 0 else 25
        if player_age >= 32:
            features['age_injury_factor'] = 1.2  # 20% higher risk
        else:
            features['age_injury_factor'] = 1.0

        return features

    def generate_all_external_features(self, player_name, game_date, player_team, opponent_team,
                                       last_game_date=None, game_sequence='unknown'):
        """Generate all external factor features."""

        features = {}

        # Travel impact
        travel = self.calculate_travel_impact(player_name, game_date, player_team, opponent_team, last_game_date)
        features.update({f'travel_{k}': v for k, v in travel.items()})

        # Back-to-back impact
        b2b = self.calculate_back_to_back_impact(player_name, game_date, game_sequence)
        features.update({f'b2b_{k}': v for k, v in b2b.items()})

        # Team context
        team_ctx = self.calculate_team_context_features(player_name, player_team, opponent_team, game_date)
        features.update({f'team_{k}': v for k, v in team_ctx.items()})

        # Injury impact
        injury = self.estimate_injury_impact(player_name, player_team, game_date)
        features.update({f'injury_{k}': v for k, v in injury.items()})

        # Composite scoring adjustment
        features['external_scoring_adjustment'] = (
            -travel.get('travel_impact_penalty', 0) * 20 -  # Convert % to points
            -travel.get('tz_impact_penalty', 0) * 20 +
            b2b.get('b2b_impact', 0) +
            team_ctx.get('pace_factor', 1.0) * 2 -  # 2 points per pace unit
            -injury.get('injury_risk', 0) * 10 -  # Penalty for injury risk
            -injury.get('minutes_restriction', 0) * 0.5  # Points per minute lost
        )

        return features

    # Helper methods
    def _get_last_game_date(self, player_name, before_date):
        """Get the date of the player's last game before the specified date."""
        player_games = self.data[
            (self.data['fullName'] == player_name) &
            (self.data['gameDate'] < pd.to_datetime(before_date))
        ]
        if len(player_games) > 0:
            return player_games['gameDate'].max()
        return None

    def _estimate_travel_hours(self, from_team, to_team):
        """Estimate travel hours between NBA cities (simplified)."""
        # Simplified distance matrix (in hours)
        distances = {
            ('West', 'West'): 2.5,
            ('West', 'Central'): 3.5,
            ('West', 'East'): 5.0,
            ('Central', 'West'): 3.5,
            ('Central', 'Central'): 1.5,
            ('Central', 'East'): 2.5,
            ('East', 'West'): 5.0,
            ('East', 'Central'): 2.5,
            ('East', 'East'): 1.5
        }

        from_division = self._get_division(from_team)
        to_division = self._get_division(to_team)

        return distances.get((from_division, to_division), 3.0)

    def _get_division(self, team):
        """Get division of team."""
        east = ['Atlanta', 'Boston', 'Brooklyn', 'Charlotte', 'Chicago', 'Cleveland', 'Detroit',
                'Indiana', 'Miami', 'Milwaukee', 'New York', 'Orlando', 'Philadelphia', 'Toronto']
        west = ['Golden State', 'LA Clippers', 'LA Lakers', 'Phoenix', 'Sacramento', 'Portland', 'Seattle',
                'Denver', 'Minnesota', 'Oklahoma City', 'Portland', 'Utah', 'Dallas', 'Houston',
                'Memphis', 'New Orleans', 'San Antonio']

        # Extract city from team name
        for city in east:
            if city in team:
                return 'East'
        for city in west:
            if city in team:
                return 'West'
        return 'Central'  # Default

    def _get_timezone_change(self, from_team, to_team):
        """Get timezone change between teams."""
        tz_map = {'EST': 0, 'CST': 1, 'MST': 2, 'PST': 3}

        from_tz = self.team_locations.get(from_team, {}).get('timezone', 'EST')
        to_tz = self.team_locations.get(to_team, {}).get('timezone', 'EST')

        return tz_map.get(to_tz, 0) - tz_map.get(from_tz, 0)

    def _get_team_recent_performance(self, team, game_date, days_back=10):
        """Calculate team's recent winning percentage."""
        team_games = self.data[
            (self.data['playerteamName'] == team) | (self.data['opponentteamName'] == team)
        ]

        cutoff_date = pd.to_datetime(game_date) - timedelta(days=days_back)
        recent_games = team_games[team_games['gameDate'] >= cutoff_date]

        if len(recent_games) > 0:
            # Simplified: use points scored as proxy for performance
            return recent_games['points'].mean() / 100
        return 0.5

    def _empty_travel_features(self):
        """Return empty travel features."""
        return {
            'travel_hours': 0,
            'travel_impact_penalty': 0,
            'timezone_change': 0,
            'tz_impact_penalty': 0,
            'is_back_to_back': False
        }

# Test external factors
if __name__ == "__main__":
    print("Testing External Factors - Iteration 1")
    print("=" * 40)

    try:
        # Load data
        data = pd.read_csv('../data/processed/engineered_features.csv')
        generator = ExternalFactorsV1(data)

        # Test
        features = generator.generate_all_external_features(
            'LeBron James',
            datetime.now().strftime('%Y-%m-%d'),
            'Los Angeles Lakers',
            'Golden State Warriors'
        )

        print(f"Generated {len(features)} external features:")
        for feature, value in features.items():
            if isinstance(value, float):
                print(f"  {feature}: {value:.2f}")
            else:
                print(f"  {feature}: {value}")

    except Exception as e:
        print(f"Error: {e}")