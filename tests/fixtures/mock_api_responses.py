"""
Mock API Responses for Testing
===============================

Provides realistic mock responses from The Odds API for testing
without consuming API credits.
"""

from datetime import datetime, timedelta
import json


class MockOddsAPIResponses:
    """Collection of mock API responses for testing."""

    @staticmethod
    def get_games_response(num_games=3):
        """Generate mock response for games endpoint."""
        base_time = datetime.now() + timedelta(hours=3)

        games = []
        teams = [
            ('Los Angeles Lakers', 'Golden State Warriors'),
            ('Boston Celtics', 'Miami Heat'),
            ('Phoenix Suns', 'Dallas Mavericks'),
            ('Milwaukee Bucks', 'Brooklyn Nets'),
            ('Denver Nuggets', 'Los Angeles Clippers'),
        ]

        for i in range(min(num_games, len(teams))):
            game_time = base_time + timedelta(hours=i * 2)
            home_team, away_team = teams[i]

            games.append({
                'id': f'game_{i + 1:03d}',
                'sport_key': 'basketball_nba',
                'sport_title': 'NBA',
                'commence_time': game_time.isoformat() + 'Z',
                'home_team': home_team,
                'away_team': away_team,
                'bookmakers': [
                    {
                        'key': 'draftkings',
                        'title': 'DraftKings',
                        'last_update': datetime.now().isoformat() + 'Z',
                        'markets': [
                            {
                                'key': 'h2h',
                                'outcomes': [
                                    {
                                        'name': home_team,
                                        'price': -150
                                    },
                                    {
                                        'name': away_team,
                                        'price': 130
                                    }
                                ]
                            }
                        ]
                    }
                ]
            })

        return games

    @staticmethod
    def get_player_props_response(game_id='game_001'):
        """Generate mock response for player props endpoint."""
        return {
            'id': game_id,
            'sport_key': 'basketball_nba',
            'sport_title': 'NBA',
            'commence_time': (datetime.now() + timedelta(hours=3)).isoformat() + 'Z',
            'home_team': 'Los Angeles Lakers',
            'away_team': 'Golden State Warriors',
            'bookmakers': [
                {
                    'key': 'draftkings',
                    'title': 'DraftKings',
                    'last_update': datetime.now().isoformat() + 'Z',
                    'markets': [
                        {
                            'key': 'player_points',
                            'outcomes': [
                                {
                                    'name': 'Over',
                                    'description': 'LeBron James',
                                    'price': -110,
                                    'point': 25.5
                                },
                                {
                                    'name': 'Under',
                                    'description': 'LeBron James',
                                    'price': -110,
                                    'point': 25.5
                                },
                                {
                                    'name': 'Over',
                                    'description': 'Stephen Curry',
                                    'price': -115,
                                    'point': 28.5
                                },
                                {
                                    'name': 'Under',
                                    'description': 'Stephen Curry',
                                    'price': -105,
                                    'point': 28.5
                                },
                                {
                                    'name': 'Over',
                                    'description': 'Anthony Davis',
                                    'price': -110,
                                    'point': 24.5
                                },
                                {
                                    'name': 'Under',
                                    'description': 'Anthony Davis',
                                    'price': -110,
                                    'point': 24.5
                                }
                            ]
                        },
                        {
                            'key': 'player_assists',
                            'outcomes': [
                                {
                                    'name': 'Over',
                                    'description': 'LeBron James',
                                    'price': -105,
                                    'point': 7.5
                                },
                                {
                                    'name': 'Under',
                                    'description': 'LeBron James',
                                    'price': -115,
                                    'point': 7.5
                                },
                                {
                                    'name': 'Over',
                                    'description': 'Stephen Curry',
                                    'price': -110,
                                    'point': 6.5
                                },
                                {
                                    'name': 'Under',
                                    'description': 'Stephen Curry',
                                    'price': -110,
                                    'point': 6.5
                                }
                            ]
                        },
                        {
                            'key': 'player_rebounds',
                            'outcomes': [
                                {
                                    'name': 'Over',
                                    'description': 'Anthony Davis',
                                    'price': -110,
                                    'point': 11.5
                                },
                                {
                                    'name': 'Under',
                                    'description': 'Anthony Davis',
                                    'price': -110,
                                    'point': 11.5
                                },
                                {
                                    'name': 'Over',
                                    'description': 'LeBron James',
                                    'price': -105,
                                    'point': 8.5
                                },
                                {
                                    'name': 'Under',
                                    'description': 'LeBron James',
                                    'price': -115,
                                    'point': 8.5
                                }
                            ]
                        }
                    ]
                },
                {
                    'key': 'fanduel',
                    'title': 'FanDuel',
                    'last_update': datetime.now().isoformat() + 'Z',
                    'markets': [
                        {
                            'key': 'player_points',
                            'outcomes': [
                                {
                                    'name': 'Over',
                                    'description': 'LeBron James',
                                    'price': -108,
                                    'point': 26.5
                                },
                                {
                                    'name': 'Under',
                                    'description': 'LeBron James',
                                    'price': -112,
                                    'point': 26.5
                                },
                                {
                                    'name': 'Over',
                                    'description': 'Stephen Curry',
                                    'price': -110,
                                    'point': 29.5
                                },
                                {
                                    'name': 'Under',
                                    'description': 'Stephen Curry',
                                    'price': -110,
                                    'point': 29.5
                                }
                            ]
                        }
                    ]
                }
            ]
        }

    @staticmethod
    def get_empty_response():
        """Generate empty response (no games available)."""
        return []

    @staticmethod
    def get_error_response():
        """Generate error response."""
        return {
            'success': False,
            'error': 'Invalid API key'
        }

    @staticmethod
    def get_rate_limit_headers(remaining=100):
        """Generate mock rate limit headers."""
        return {
            'x-requests-remaining': str(remaining),
            'x-requests-used': str(500 - remaining)
        }


class MockAPIClient:
    """Mock API client for testing without making real requests."""

    def __init__(self, api_key='test_key', fail_requests=False, low_rate_limit=False):
        """
        Initialize mock client.

        Args:
            api_key: Mock API key
            fail_requests: If True, simulate failed requests
            low_rate_limit: If True, simulate low remaining requests
        """
        self.api_key = api_key
        self.fail_requests = fail_requests
        self.low_rate_limit = low_rate_limit
        self.request_count = 0

    def get(self, url, params=None, timeout=None):
        """Mock GET request."""
        self.request_count += 1

        # Simulate failed request
        if self.fail_requests:
            raise ConnectionError("Failed to connect to API")

        # Create mock response
        response = MockResponse()

        # Set headers
        remaining = 5 if self.low_rate_limit else 100
        response.headers = MockOddsAPIResponses.get_rate_limit_headers(remaining)

        # Determine endpoint and return appropriate data
        if 'odds' in url and 'events' not in url:
            response._json_data = MockOddsAPIResponses.get_games_response()
        elif 'events' in url:
            response._json_data = MockOddsAPIResponses.get_player_props_response()
        else:
            response._json_data = MockOddsAPIResponses.get_empty_response()

        return response


class MockResponse:
    """Mock HTTP response object."""

    def __init__(self):
        self._json_data = None
        self.headers = {}
        self.status_code = 200
        self.text = ''

    def json(self):
        """Return JSON data."""
        if self._json_data is None:
            raise json.JSONDecodeError("No JSON data", "", 0)
        return self._json_data

    def raise_for_status(self):
        """Raise exception for bad status codes."""
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def create_mock_api_session(scenario='normal'):
    """
    Create a mock API session for different scenarios.

    Args:
        scenario: One of 'normal', 'no_games', 'rate_limit', 'error'

    Returns:
        MockAPIClient configured for the scenario
    """
    scenarios = {
        'normal': MockAPIClient(),
        'no_games': MockAPIClient(),
        'rate_limit': MockAPIClient(low_rate_limit=True),
        'error': MockAPIClient(fail_requests=True),
    }

    return scenarios.get(scenario, MockAPIClient())


# Sample player data for testing
SAMPLE_TRACKED_PLAYERS = [
    'LeBron James',
    'Stephen Curry',
    'Kevin Durant',
    'Giannis Antetokounmpo',
    'Luka Doncic',
    'Nikola Jokic',
    'Joel Embiid',
    'Anthony Davis',
    'Damian Lillard',
    'Jayson Tatum'
]


def get_sample_historical_data():
    """Get sample historical player data for testing."""
    import pandas as pd
    import numpy as np

    np.random.seed(42)

    data = []
    for player in SAMPLE_TRACKED_PLAYERS[:5]:  # Use subset for faster tests
        dates = pd.date_range('2023-10-01', '2024-01-15', freq='2D')

        for date in dates:
            data.append({
                'gameDate': date,
                'fullName': player,
                'points': np.random.randint(15, 40),
                'assists': np.random.randint(2, 12),
                'reboundsTotal': np.random.randint(3, 15),
                'numMinutes': np.random.randint(25, 42),
                'rolling_3g_points': np.random.uniform(15, 35),
                'rolling_5g_points': np.random.uniform(18, 33),
                'rolling_3g_assists': np.random.uniform(2, 10),
                'age': np.random.randint(24, 36),
                'player_threshold': np.random.uniform(20, 30),
                'over_threshold': np.random.randint(0, 2)
            })

    return pd.DataFrame(data)


if __name__ == '__main__':
    # Test the mock responses
    print("Testing Mock API Responses")
    print("=" * 50)

    print("\n1. Games Response:")
    games = MockOddsAPIResponses.get_games_response(2)
    print(f"   Generated {len(games)} games")

    print("\n2. Player Props Response:")
    props = MockOddsAPIResponses.get_player_props_response()
    print(f"   Game ID: {props['id']}")
    print(f"   Bookmakers: {len(props['bookmakers'])}")

    print("\n3. Mock API Client:")
    client = MockAPIClient()
    response = client.get('https://api.example.com/odds')
    print(f"   Status: {response.status_code}")
    print(f"   Requests made: {client.request_count}")

    print("\n4. Sample Historical Data:")
    historical = get_sample_historical_data()
    print(f"   Shape: {historical.shape}")
    print(f"   Players: {historical['fullName'].nunique()}")

    print("\nAll mock utilities working correctly!")
