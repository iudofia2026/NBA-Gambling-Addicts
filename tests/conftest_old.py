"""
Pytest Configuration and Shared Fixtures
=========================================

Global pytest configuration and reusable fixtures for all tests.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ========================================
# SESSION FIXTURES (Setup once per test session)
# ========================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="nba_test_")
    yield temp_dir
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_player_data():
    """Generate sample player data for testing."""
    np.random.seed(42)

    players = ['LeBron James', 'Stephen Curry', 'Kevin Durant']
    teams = ['LAL', 'GSW', 'PHX']

    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='2D')

    data = []
    for player in players:
        for date in dates[:50]:  # 50 games per player
            data.append({
                'gameDate': date,
                'fullName': player,
                'teamAbbreviation': np.random.choice(teams),
                'points': np.random.randint(15, 40),
                'assists': np.random.randint(2, 12),
                'reboundsTotal': np.random.randint(3, 15),
                'numMinutes': np.random.randint(25, 42),
                'fieldGoalsMade': np.random.randint(6, 15),
                'fieldGoalsAttempted': np.random.randint(12, 25),
                'threePointersMade': np.random.randint(0, 6),
                'threePointersAttempted': np.random.randint(2, 12),
                'freeThrowsMade': np.random.randint(2, 10),
                'freeThrowsAttempted': np.random.randint(3, 12),
                'steals': np.random.randint(0, 3),
                'blocks': np.random.randint(0, 2),
                'turnovers': np.random.randint(1, 5),
                'age': np.random.randint(24, 36),
            })

    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def sample_features_data(sample_player_data):
    """Generate sample feature-engineered data."""
    df = sample_player_data.copy()

    # Add engineered features
    df['rolling_3g_points'] = df.groupby('fullName')['points'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    )
    df['rolling_5g_points'] = df.groupby('fullName')['points'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
    )
    df['rolling_3g_assists'] = df.groupby('fullName')['assists'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    )

    # Add target variable
    df['player_threshold'] = df.groupby('fullName')['points'].transform('median')
    df['over_threshold'] = (df['points'] > df['player_threshold']).astype(int)

    return df.fillna(0)


# ========================================
# FUNCTION FIXTURES (Setup per test function)
# ========================================

@pytest.fixture
def mock_api_key():
    """Mock API key for testing."""
    return "test_api_key_12345"


@pytest.fixture
def mock_odds_response():
    """Mock response from The Odds API for games."""
    return [
        {
            'id': 'game_123',
            'home_team': 'Los Angeles Lakers',
            'away_team': 'Golden State Warriors',
            'commence_time': '2024-01-15T19:00:00Z',
            'bookmakers': [
                {
                    'title': 'DraftKings',
                    'markets': [
                        {
                            'key': 'h2h',
                            'outcomes': [
                                {'name': 'Los Angeles Lakers', 'price': -150},
                                {'name': 'Golden State Warriors', 'price': 130}
                            ]
                        }
                    ]
                }
            ]
        }
    ]


@pytest.fixture
def mock_player_props_response():
    """Mock response for player props."""
    return {
        'id': 'game_123',
        'home_team': 'Los Angeles Lakers',
        'away_team': 'Golden State Warriors',
        'bookmakers': [
            {
                'title': 'DraftKings',
                'markets': [
                    {
                        'key': 'player_points',
                        'outcomes': [
                            {
                                'description': 'LeBron James',
                                'point': 25.5,
                                'price': -110
                            },
                            {
                                'description': 'Stephen Curry',
                                'point': 28.5,
                                'price': -115
                            }
                        ]
                    },
                    {
                        'key': 'player_assists',
                        'outcomes': [
                            {
                                'description': 'LeBron James',
                                'point': 7.5,
                                'price': -105
                            }
                        ]
                    }
                ]
            }
        ]
    }


@pytest.fixture
def mock_trained_model():
    """Mock trained ML model."""
    model = MagicMock()
    model.predict.return_value = np.array([1, 0, 1])
    model.predict_proba.return_value = np.array([
        [0.3, 0.7],
        [0.6, 0.4],
        [0.2, 0.8]
    ])
    return model


@pytest.fixture
def sample_prediction_input():
    """Sample input data for predictions."""
    return pd.DataFrame({
        'fullName': ['LeBron James', 'Stephen Curry'],
        'gameDate': pd.to_datetime(['2024-01-15', '2024-01-15']),
        'home_team': ['Los Angeles Lakers', 'Golden State Warriors'],
        'away_team': ['Golden State Warriors', 'Phoenix Suns'],
        'prop_line': [25.5, 28.5],
        'market_type': ['points', 'points'],
        'over_odds': [-110, -115],
        'bookmaker': ['DraftKings', 'DraftKings'],
        'game_time': [datetime.now(), datetime.now()]
    })


@pytest.fixture
def temp_model_dir(test_data_dir):
    """Create temporary directory for model files."""
    model_dir = os.path.join(test_data_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


@pytest.fixture
def temp_data_dir(test_data_dir):
    """Create temporary directory for data files."""
    data_dir = os.path.join(test_data_dir, 'data', 'processed')
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


# ========================================
# MOCK ENVIRONMENT FIXTURES
# ========================================

@pytest.fixture
def mock_env_with_api_key(monkeypatch, mock_api_key):
    """Set up environment with API key."""
    monkeypatch.setenv('ODDS_API_KEY', mock_api_key)
    yield mock_api_key
    monkeypatch.delenv('ODDS_API_KEY', raising=False)


@pytest.fixture
def mock_env_without_api_key(monkeypatch):
    """Set up environment without API key."""
    monkeypatch.delenv('ODDS_API_KEY', raising=False)


# ========================================
# PERFORMANCE MARKERS
# ========================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require API mocking"
    )


# ========================================
# PYTEST HOOKS
# ========================================

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Add integration marker to tests in integration/ directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark slow tests that take >2 seconds
        if "slow" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
