"""
Mock API Responses for Optimized NBA Predictions System
======================================================

Provides realistic mock responses for external dependencies including
Odds API, model predictions, and data sources. Ensures tests don't
require external services while maintaining realistic behavior.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np


class MockOddsAPIResponses:
    """Mock responses for The Odds API endpoints."""

    @staticmethod
    def get_todays_games_response() -> List[Dict[str, Any]]:
        """Mock response for today's NBA games."""
        return [
            {
                'id': 'game_lal_gsw_20240115',
                'sport_key': 'basketball_nba',
                'sport_title': 'NBA',
                'commence_time': '2024-01-15T19:00:00Z',
                'home_team': 'Los Angeles Lakers',
                'away_team': 'Golden State Warriors',
                'bookmakers': [
                    {
                        'key': 'draftkings',
                        'title': 'DraftKings',
                        'last_update': '2024-01-15T12:00:00Z',
                        'markets': [
                            {
                                'key': 'h2h',
                                'last_update': '2024-01-15T12:00:00Z',
                                'outcomes': [
                                    {
                                        'name': 'Los Angeles Lakers',
                                        'price': -110
                                    },
                                    {
                                        'name': 'Golden State Warriors',
                                        'price': -110
                                    }
                                ]
                            }
                        ]
                    }
                ]
            },
            {
                'id': 'game_phx_bos_20240115',
                'sport_key': 'basketball_nba',
                'sport_title': 'NBA',
                'commence_time': '2024-01-15T19:30:00Z',
                'home_team': 'Phoenix Suns',
                'away_team': 'Boston Celtics',
                'bookmakers': [
                    {
                        'key': 'fanduel',
                        'title': 'FanDuel',
                        'last_update': '2024-01-15T12:00:00Z',
                        'markets': [
                            {
                                'key': 'h2h',
                                'last_update': '2024-01-15T12:00:00Z',
                                'outcomes': [
                                    {
                                        'name': 'Phoenix Suns',
                                        'price': -105
                                    },
                                    {
                                        'name': 'Boston Celtics',
                                        'price': -115
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ]

    @staticmethod
    def get_player_props_response(game_id: str = None) -> List[Dict[str, Any]]:
        """Mock response for player props by game."""
        base_game_id = game_id or 'game_lal_gsw_20240115'

        responses = {
            'game_lal_gsw_20240115': {
                'id': 'game_lal_gsw_20240115',
                'sport_key': 'basketball_nba',
                'sport_title': 'NBA',
                'commence_time': '2024-01-15T19:00:00Z',
                'home_team': 'Los Angeles Lakers',
                'away_team': 'Golden State Warriors',
                'bookmakers': [
                    {
                        'key': 'draftkings',
                        'title': 'DraftKings',
                        'last_update': '2024-01-15T12:00:00Z',
                        'markets': [
                            {
                                'key': 'player_points',
                                'last_update': '2024-01-15T12:00:00Z',
                                'outcomes': [
                                    {
                                        'description': 'LeBron James',
                                        'name': 'LeBron James',
                                        'price': -110,
                                        'point': 25.5
                                    },
                                    {
                                        'description': 'Stephen Curry',
                                        'name': 'Stephen Curry',
                                        'price': -115,
                                        'point': 28.5
                                    },
                                    {
                                        'description': 'Anthony Davis',
                                        'name': 'Anthony Davis',
                                        'price': -105,
                                        'point': 22.5
                                    }
                                ]
                            },
                            {
                                'key': 'player_rebounds',
                                'last_update': '2024-01-15T12:00:00Z',
                                'outcomes': [
                                    {
                                        'description': 'LeBron James',
                                        'name': 'LeBron James',
                                        'price': -110,
                                        'point': 8.5
                                    },
                                    {
                                        'description': 'Anthony Davis',
                                        'name': 'Anthony Davis',
                                        'price': -115,
                                        'point': 11.5
                                    }
                                ]
                            },
                            {
                                'key': 'player_assists',
                                'last_update': '2024-01-15T12:00:00Z',
                                'outcomes': [
                                    {
                                        'description': 'LeBron James',
                                        'name': 'LeBron James',
                                        'price': -105,
                                        'point': 7.5
                                    },
                                    {
                                        'description': 'Stephen Curry',
                                        'name': 'Stephen Curry',
                                        'price': -110,
                                        'point': 6.5
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        'key': 'fanduel',
                        'title': 'FanDuel',
                        'last_update': '2024-01-15T12:00:00Z',
                        'markets': [
                            {
                                'key': 'player_points',
                                'last_update': '2024-01-15T12:00:00Z',
                                'outcomes': [
                                    {
                                        'description': 'LeBron James',
                                        'name': 'LeBron James',
                                        'price': -108,
                                        'point': 26.0  # Different line
                                    }
                                ]
                            }
                        ]
                    }
                ]
            },
            'game_phx_bos_20240115': {
                'id': 'game_phx_bos_20240115',
                'sport_key': 'basketball_nba',
                'sport_title': 'NBA',
                'commence_time': '2024-01-15T19:30:00Z',
                'home_team': 'Phoenix Suns',
                'away_team': 'Boston Celtics',
                'bookmakers': [
                    {
                        'key': 'draftkings',
                        'title': 'DraftKings',
                        'last_update': '2024-01-15T12:00:00Z',
                        'markets': [
                            {
                                'key': 'player_points',
                                'last_update': '2024-01-15T12:00:00Z',
                                'outcomes': [
                                    {
                                        'description': 'Kevin Durant',
                                        'name': 'Kevin Durant',
                                        'price': -110,
                                        'point': 27.5
                                    },
                                    {
                                        'description': 'Jayson Tatum',
                                        'name': 'Jayson Tatum',
                                        'price': -115,
                                        'point': 29.5
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        }

        return [responses.get(base_game_id, responses['game_lal_gsw_20240115'])]

    @staticmethod
    def get_empty_response() -> List:
        """Empty response when no games/props available."""
        return []

    @staticmethod
    def get_error_response() -> Dict[str, Any]:
        """Error response from API."""
        return {
            'error': 'Rate limit exceeded',
            'message': 'Too many requests',
            'status': 429
        }


class MockModelResponses:
    """Mock responses for ML model predictions."""

    @staticmethod
    def get_random_forest_prediction() -> Dict[str, Any]:
        """Mock Random Forest model prediction response."""
        return {
            'prediction': 1,  # 1 = OVER, 0 = UNDER
            'probabilities': [0.3, 0.7],  # [P(UNDER), P(OVER)]
            'feature_importance': {
                'rolling_3g_points': 0.25,
                'rolling_5g_points': 0.20,
                'opp_defensive_rating': 0.15,
                'player_usage_rate': 0.12,
                'days_rest': 0.08,
                'home_advantage': 0.05,
                'back_to_back': 0.03,
                'injury_status': 0.02,
                'team_momentum': 0.10
            }
        }

    @staticmethod
    def get_xgboost_prediction() -> Dict[str, Any]:
        """Mock XGBoost model prediction response."""
        return {
            'prediction': 0,  # UNDER
            'probabilities': [0.65, 0.35],  # [P(UNDER), P(OVER)]
            'feature_importance': {
                'rolling_3g_points': 0.22,
                'rolling_5g_points': 0.18,
                'opp_defensive_rating': 0.16,
                'player_usage_rate': 0.14,
                'days_rest': 0.10,
                'home_advantage': 0.06,
                'back_to_back': 0.04,
                'injury_status': 0.01,
                'team_momentum': 0.09
            }
        }

    @staticmethod
    def get_ensemble_prediction() -> Dict[str, Any]:
        """Mock ensemble prediction combining multiple models."""
        rf_pred = MockModelResponses.get_random_forest_prediction()
        xgb_pred = MockModelResponses.get_xgboost_prediction()

        # Average probabilities
        avg_prob_under = (rf_pred['probabilities'][0] + xgb_pred['probabilities'][0]) / 2
        avg_prob_over = (rf_pred['probabilities'][1] + xgb_pred['probabilities'][1]) / 2

        # Final prediction based on higher probability
        final_prediction = 1 if avg_prob_over > avg_prob_under else 0

        return {
            'prediction': final_prediction,
            'probabilities': [avg_prob_under, avg_prob_over],
            'model_contributions': {
                'random_forest': rf_pred,
                'xgboost': xgb_pred
            },
            'confidence': max(avg_prob_under, avg_prob_over)
        }


class MockHistoricalData:
    """Mock historical player data for testing."""

    @staticmethod
    def get_sample_historical_data(days_back: int = 30) -> List[Dict[str, Any]]:
        """Generate sample historical data for a player."""
        base_date = datetime.now() - timedelta(days=days_back)
        players = ['LeBron James', 'Stephen Curry', 'Kevin Durant']
        teams = ['LAL', 'GSW', 'PHX', 'BOS', 'MIA']

        data = []
        for i, player in enumerate(players):
            for day in range(days_back):
                game_date = base_date + timedelta(days=day)

                # Generate realistic stats with some variance
                base_points = {'LeBron James': 27, 'Stephen Curry': 29, 'Kevin Durant': 26}[player]
                points = max(10, base_points + np.random.randint(-10, 10))

                base_minutes = {'LeBron James': 35, 'Stephen Curry': 34, 'Kevin Durant': 36}[player]
                minutes = max(20, base_minutes + np.random.randint(-5, 5))

                data.append({
                    'gameDate': game_date.strftime('%Y-%m-%d'),
                    'fullName': player,
                    'playerteamName': teams[i % len(teams)],
                    'opponentteamName': teams[(i + 1) % len(teams)],
                    'points': points,
                    'assists': max(1, int(points * 0.25 + np.random.randint(-3, 3))),
                    'reboundsTotal': max(1, int(points * 0.35 + np.random.randint(-3, 3))),
                    'numMinutes': minutes,
                    'fieldGoalsMade': max(1, int(points * 0.45 + np.random.randint(-2, 2))),
                    'fieldGoalsAttempted': max(5, int(points * 0.85 + np.random.randint(-3, 3))),
                    'threePointersMade': max(0, int(points * 0.15 + np.random.randint(-2, 2))),
                    'threePointersAttempted': max(2, int(points * 0.35 + np.random.randint(-2, 2))),
                    'freeThrowsMade': max(0, int(points * 0.20 + np.random.randint(-2, 2))),
                    'freeThrowsAttempted': max(1, int(points * 0.25 + np.random.randint(-2, 2))),
                    'steals': max(0, np.random.randint(0, 4)),
                    'blocks': max(0, np.random.randint(0, 3)),
                    'turnovers': max(1, np.random.randint(1, 6)),
                    'age': {'LeBron James': 38, 'Stephen Curry': 35, 'Kevin Durant': 34}[player],
                    'over_threshold': 1 if points > 25 else 0  # Simple threshold
                })

        return data

    @staticmethod
    def get_engineered_features_data() -> List[Dict[str, Any]]:
        """Generate sample engineered features data."""
        historical_data = MockHistoricalData.get_sample_historical_data()

        # Convert to DataFrame and add engineered features
        import pandas as pd
        df = pd.DataFrame(historical_data)
        df['gameDate'] = pd.to_datetime(df['gameDate'])

        # Sort by player and date for rolling calculations
        df = df.sort_values(['fullName', 'gameDate'])

        # Add rolling features
        df['rolling_3g_points'] = df.groupby('fullName')['points'].transform(
            lambda x: x.rolling(3, min_periods=1).mean().shift(1)
        ).fillna(df['points'].mean())

        df['rolling_5g_points'] = df.groupby('fullName')['points'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        ).fillna(df['points'].mean())

        df['rolling_3g_assists'] = df.groupby('fullName')['assists'].transform(
            lambda x: x.rolling(3, min_periods=1).mean().shift(1)
        ).fillna(0)

        df['rolling_3g_rebounds'] = df.groupby('fullName')['reboundsTotal'].transform(
            lambda x: x.rolling(3, min_periods=1).mean().shift(1)
        ).fillna(0)

        # Add efficiency features
        df['points_per_minute'] = (df['points'] / df['numMinutes']).fillna(0)
        df['true_shooting_pct'] = (df['points'] / (2 * (df['fieldGoalsAttempted'] + 0.44 * df['freeThrowsAttempted']))).fillna(0)

        # Add opponent strength (mock rating)
        opponent_ratings = {'LAL': 115.2, 'GSW': 114.8, 'PHX': 116.1, 'BOS': 117.3, 'MIA': 112.9}
        df['opp_defensive_rating'] = df['opponentteamName'].map(opponent_ratings).fillna(114.0)

        # Add situational features
        df['days_rest'] = df.groupby('fullName')['gameDate'].transform(
            lambda x: x.diff().dt.days.fillna(3)
        ).fillna(3)

        df['home_game'] = df['playerteamName'].isin(['LAL', 'GSW', 'PHX', 'BOS', 'MIA']).astype(int)

        return df.to_dict('records')


class MockPredictionResults:
    """Mock prediction results for testing."""

    @staticmethod
    def get_sample_predictions() -> List[Dict[str, Any]]:
        """Generate sample prediction results."""
        return [
            {
                'player_name': 'LeBron James',
                'market_type': 'points',
                'prop_line': 25.5,
                'recommendation': 'OVER',
                'confidence': 0.78,
                'over_odds': -110,
                'bookmaker': 'DraftKings',
                'game_time': datetime.now() + timedelta(hours=3),
                'optimized_insights': {
                    'predicted_value': 28.2,
                    'line_diff': 2.7,
                    'confidence_level': 'High',
                    'form': 'hot (82%)',
                    'matchup_quality': '73%',
                    'team_chemistry': '79%',
                    'adjustment': '+2.1'
                }
            },
            {
                'player_name': 'Stephen Curry',
                'market_type': 'points',
                'prop_line': 28.5,
                'recommendation': 'UNDER',
                'confidence': 0.71,
                'over_odds': -115,
                'bookmaker': 'FanDuel',
                'game_time': datetime.now() + timedelta(hours=3),
                'optimized_insights': {
                    'predicted_value': 26.8,
                    'line_diff': -1.7,
                    'confidence_level': 'Medium-High',
                    'form': 'neutral (68%)',
                    'matchup_quality': '65%',
                    'team_chemistry': '71%',
                    'adjustment': '-1.2'
                }
            },
            {
                'player_name': 'Anthony Davis',
                'market_type': 'rebounds',
                'prop_line': 11.5,
                'recommendation': 'OVER',
                'confidence': 0.74,
                'over_odds': -105,
                'bookmaker': 'DraftKings',
                'game_time': datetime.now() + timedelta(hours=3),
                'optimized_insights': {
                    'predicted_value': 13.1,
                    'line_diff': 1.6,
                    'confidence_level': 'High',
                    'form': 'hot (75%)',
                    'matchup_quality': '81%',
                    'team_chemistry': '72%',
                    'adjustment': '+1.4'
                }
            }
        ]

    @staticmethod
    def get_empty_predictions() -> List:
        """Empty predictions list."""
        return []

    @staticmethod
    def get_low_confidence_predictions() -> List[Dict[str, Any]]:
        """Predictions with low confidence (should be filtered out)."""
        return [
            {
                'player_name': 'Unknown Player',
                'market_type': 'points',
                'prop_line': 20.5,
                'recommendation': 'OVER',
                'confidence': 0.45,  # Below threshold
                'over_odds': -110,
                'bookmaker': 'DraftKings',
                'game_time': datetime.now() + timedelta(hours=5),
                'optimized_insights': {
                    'predicted_value': 19.2,
                    'line_diff': -1.3,
                    'confidence_level': 'Low',
                    'form': 'cold (32%)',
                    'matchup_quality': '41%',
                    'team_chemistry': '38%',
                    'adjustment': '-0.8'
                }
            }
        ]


class MockAPIResponses:
    """Collection of all mock API responses."""

    ODDS_API = MockOddsAPIResponses()
    MODEL = MockModelResponses()
    HISTORICAL = MockHistoricalData()
    PREDICTIONS = MockPredictionResults()

    @classmethod
    def get_all_todays_props(cls) -> List[Dict[str, Any]]:
        """Get comprehensive mock props data for all today's games."""
        games = cls.ODDS_API.get_todays_games_response()
        all_props = []

        for game in games:
            game_props = cls.ODDS_API.get_player_props_response(game['id'])
            all_props.extend(game_props)

        return all_props

    @classmethod
    def get_tracked_players(cls) -> List[str]:
        """Get list of tracked players."""
        return [
            'LeBron James', 'Stephen Curry', 'Kevin Durant', 'Anthony Davis',
            'Jayson Tatum', 'Jaylen Brown', 'Giannis Antetokounmpo', 'Damian Lillard',
            'Joel Embiid', 'Nikola Jokić', 'Luka Dončić', 'Kawhi Leonard'
        ]


# Utility functions for test setup
def create_mock_odds_client():
    """Create a mock odds client with all responses."""
    from unittest.mock import Mock

    mock_client = Mock()
    mock_client.get_todays_nba_games.return_value = MockOddsAPIResponses.get_todays_games_response()
    mock_client.get_all_todays_player_props.return_value = MockAPIResponses.get_all_todays_props()
    mock_client.get_player_props_for_game.return_value = MockOddsAPIResponses.get_player_props_response()
    mock_client.format_for_ml_pipeline.return_value = pd.DataFrame([
        {
            'fullName': 'LeBron James',
            'market_type': 'points',
            'prop_line': 25.5,
            'over_odds': -110,
            'bookmaker': 'DraftKings',
            'home_team': 'Los Angeles Lakers',
            'away_team': 'Golden State Warriors',
            'game_time': datetime.now() + timedelta(hours=3),
            'gameDate': (datetime.now() + timedelta(hours=3)).date(),
            'playerteamName': 'LAL'
        }
    ])

    return mock_client


def create_mock_models():
    """Create mock ML models."""
    from unittest.mock import Mock, MagicMock

    rf_model = MagicMock()
    rf_model.predict.return_value = np.array([1, 0, 1])
    rf_model.predict_proba.return_value = np.array([
        [0.3, 0.7],
        [0.6, 0.4],
        [0.2, 0.8]
    ])

    xgb_model = MagicMock()
    xgb_model.predict.return_value = np.array([1, 0, 1])
    xgb_model.predict_proba.return_value = np.array([
        [0.25, 0.75],
        [0.65, 0.35],
        [0.15, 0.85]
    ])

    return {
        'random_forest': rf_model,
        'xgboost': xgb_model
    }