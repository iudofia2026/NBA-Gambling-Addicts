"""
Updated Pytest Configuration and Shared Fixtures
===============================================

Global pytest configuration and reusable fixtures for all tests.
Optimized for the production system: final_predictions_optimized.py
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil
import joblib

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import optimized system for testing
from final_predictions_optimized import OptimizedNBAPredictionsSystem

# Import mock utilities
from tests.fixtures.mock_optimized_responses import (
    MockAPIResponses,
    create_mock_odds_client,
    create_mock_models
)


# ========================================
# SESSION FIXTURES (Setup once per test session)
# ========================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="nba_optimized_test_")
    yield temp_dir
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def mock_api_key():
    """Mock API key for testing."""
    return "test_optimized_api_key_12345"


@pytest.fixture(scope="session")
def sample_optimized_player_data():
    """Generate sample player data optimized for the production system."""
    np.random.seed(42)

    players = ['LeBron James', 'Stephen Curry', 'Kevin Durant', 'Anthony Davis']
    teams = ['LAL', 'GSW', 'PHX', 'BOS']
    opponents = ['GSW', 'LAL', 'PHX', 'BOS', 'MIA', 'NYK']

    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='2D')

    data = []
    for player in players:
        for date in dates[:60]:  # 60 games per player for better analysis
            # Base stats with realistic variance
            base_points = {'LeBron James': 27, 'Stephen Curry': 29, 'Kevin Durant': 26, 'Anthony Davis': 24}[player]
            points = max(10, base_points + np.random.randint(-8, 8))

            base_minutes = {'LeBron James': 35, 'Stephen Curry': 34, 'Kevin Durant': 36, 'Anthony Davis': 33}[player]
            minutes = max(20, base_minutes + np.random.randint(-5, 5))

            # Realistic stat relationships
            field_goals_made = max(5, int(points * 0.45 + np.random.randint(-2, 2)))
            field_goals_attempted = max(field_goals_made + 5, int(points * 0.85 + np.random.randint(-3, 3)))

            data.append({
                'gameDate': date,
                'fullName': player,
                'playerteamName': np.random.choice(teams),
                'opponentteamName': np.random.choice(opponents),
                'points': points,
                'assists': max(1, int(points * 0.25 + np.random.randint(-3, 3))),
                'reboundsTotal': max(1, int(points * 0.35 + np.random.randint(-3, 3))),
                'numMinutes': minutes,
                'fieldGoalsMade': field_goals_made,
                'fieldGoalsAttempted': field_goals_attempted,
                'threePointersMade': max(0, int(points * 0.15 + np.random.randint(-2, 2))),
                'threePointersAttempted': max(2, int(points * 0.35 + np.random.randint(-2, 2))),
                'freeThrowsMade': max(0, int(points * 0.20 + np.random.randint(-2, 2))),
                'freeThrowsAttempted': max(1, int(points * 0.25 + np.random.randint(-2, 2))),
                'steals': max(0, np.random.randint(0, 4)),
                'blocks': max(0, np.random.randint(0, 3)),
                'turnovers': max(1, np.random.randint(1, 6)),
                'age': {'LeBron James': 38, 'Stephen Curry': 35, 'Kevin Durant': 34, 'Anthony Davis': 30}[player],
                # Add target variables
                'over_threshold': 1 if points > 25 else 0,
                'player_threshold': 25  # Simplified threshold
            })

    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def sample_engineered_features(sample_optimized_player_data):
    """Generate sample feature-engineered data with optimized features."""
    df = sample_optimized_player_data.copy()

    # Sort for proper rolling calculations
    df = df.sort_values(['fullName', 'gameDate'])
    df['gameDate'] = pd.to_datetime(df['gameDate'])

    # Add core rolling features
    df['rolling_3g_points'] = df.groupby('fullName')['points'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    ).fillna(0)

    df['rolling_5g_points'] = df.groupby('fullName')['points'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
    ).fillna(0)

    df['rolling_3g_assists'] = df.groupby('fullName')['assists'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    ).fillna(0)

    df['rolling_3g_rebounds'] = df.groupby('fullName')['reboundsTotal'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    ).fillna(0)

    # Add efficiency features
    df['points_per_minute'] = (df['points'] / df['numMinutes']).fillna(0)
    df['usage_rate'] = (df['fieldGoalsAttempted'] + 0.44 * df['freeThrowsAttempted']) / df['numMinutes']
    df['efficiency'] = df['points'] / (df['fieldGoalsAttempted'] * 2 + df['freeThrowsAttempted'])

    # Add opponent strength (mock ratings)
    opponent_ratings = {'LAL': 115.2, 'GSW': 114.8, 'PHX': 116.1, 'BOS': 117.3, 'MIA': 112.9, 'NYK': 113.5}
    df['opp_defensive_rating'] = df['opponentteamName'].map(opponent_ratings).fillna(114.0)

    # Add situational features
    df['days_rest'] = df.groupby('fullName')['gameDate'].transform(
        lambda x: x.diff().dt.days.fillna(3)
    ).fillna(3)

    df['home_advantage'] = (df['playerteamName'].isin(['LAL', 'GSW', 'PHX', 'BOS'])).astype(int)

    # Add performance consistency
    df['points_std_3g'] = df.groupby('fullName')['points'].transform(
        lambda x: x.rolling(3, min_periods=1).std().shift(1)
    ).fillna(df['points'].std())

    return df


# ========================================
# FUNCTION FIXTURES (Setup per test function)
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


@pytest.fixture
def temp_model_dir(test_data_dir):
    """Create temporary directory for model files."""
    model_dir = os.path.join(test_data_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Create mock model files
    models = create_mock_models()
    for model_name, model in models.items():
        model_path = os.path.join(model_dir, f'{model_name}_model.pkl')
        joblib.dump(model, model_path)

    return model_dir


@pytest.fixture
def temp_data_dir(test_data_dir):
    """Create temporary directory for data files."""
    data_dir = os.path.join(test_data_dir, 'data', 'processed')
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


@pytest.fixture
def sample_prediction_input():
    """Sample input data for predictions."""
    return pd.DataFrame({
        'fullName': ['LeBron James', 'Stephen Curry', 'Kevin Durant'],
        'gameDate': [datetime.now() + timedelta(hours=3)] * 3,
        'home_team': ['Los Angeles Lakers', 'Golden State Warriors', 'Phoenix Suns'],
        'away_team': ['Golden State Warriors', 'Phoenix Suns', 'Los Angeles Lakers'],
        'prop_line': [25.5, 28.5, 27.0],
        'market_type': ['points', 'points', 'rebounds'],
        'over_odds': [-110, -115, -105],
        'bookmaker': ['DraftKings', 'FanDuel', 'DraftKings'],
        'game_time': [datetime.now() + timedelta(hours=3)] * 3,
        'playerteamName': ['LAL', 'GSW', 'PHX'],
        'opponentteamName': ['GSW', 'PHX', 'LAL']
    })


@pytest.fixture
def mock_trained_models():
    """Mock trained ML models for testing."""
    return create_mock_models()


# ========================================
# MOCK ENVIRONMENT FIXTURES
# ========================================

@pytest.fixture
def mock_odds_client():
    """Mock odds API client with realistic responses."""
    return create_mock_odds_client()


@pytest.fixture
def mock_optimized_system(mock_api_key, sample_engineered_features):
    """Create a mock optimized predictions system."""
    with patch('final_predictions_optimized.NBAOddsClient') as mock_client:
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        system.historical_data = sample_engineered_features
        yield system


@pytest.fixture
def optimized_system_with_models(mock_api_key, sample_engineered_features, temp_model_dir):
    """Create optimized system with loaded models."""
    original_cwd = os.getcwd()
    try:
        # Change to directory structure expected by the system
        os.chdir(temp_model_dir.replace('/models', ''))

        with patch('final_predictions_optimized.NBAOddsClient') as mock_client:
            system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
            system.load_models()
            yield system

    finally:
        os.chdir(original_cwd)


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
    config.addinivalue_line(
        "markers", "performance: marks tests that measure performance"
    )
    config.addinivalue_line(
        "markers", "validation: marks tests for data validation"
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

        # Add validation marker to tests in validation/ directory
        elif "validation" in str(item.fspath):
            item.add_marker(pytest.mark.validation)

        # Mark slow tests that take >2 seconds
        if "slow" in item.nodeid.lower() or "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)

        # Mark API tests
        if "api" in item.nodeid.lower() or "odds" in item.nodeid.lower():
            item.add_marker(pytest.mark.api)


# ========================================
# CUSTOM HELPERS
# ========================================

@pytest.fixture
def assert_optimized_prediction_structure():
    """Helper to assert prediction structure is correct."""
    def _assert_structure(prediction):
        required_fields = [
            'player_name', 'market_type', 'prop_line', 'recommendation',
            'confidence', 'over_odds', 'bookmaker', 'optimized_insights'
        ]

        for field in required_fields:
            assert field in prediction, f"Missing required field: {field}"

        # Validate data types and ranges
        assert isinstance(prediction['confidence'], (int, float))
        assert 0 <= prediction['confidence'] <= 1
        assert prediction['recommendation'] in ['OVER', 'UNDER']

        # Validate optimized insights
        insights = prediction['optimized_insights']
        insight_fields = [
            'predicted_value', 'line_diff', 'confidence_level',
            'form', 'matchup_quality', 'team_chemistry', 'adjustment'
        ]

        for field in insight_fields:
            assert field in insights, f"Missing insight field: {field}"

    return _assert_structure


@pytest.fixture
def create_temp_csv_file():
    """Helper to create temporary CSV files."""
    def _create_csv(data, filename, directory=None):
        if directory is None:
            directory = tempfile.mkdtemp()

        filepath = os.path.join(directory, filename)
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        else:
            pd.DataFrame(data).to_csv(filepath, index=False)

        return filepath

    return _create_csv


# ========================================
# MOCK RESPONSES FOR TESTING
# ========================================

@pytest.fixture
def mock_odds_responses():
    """Provide access to mock API responses."""
    return MockAPIResponses.ODDS_API


@pytest.fixture
def mock_model_responses():
    """Provide access to mock model responses."""
    return MockAPIResponses.MODEL


@pytest.fixture
def mock_prediction_results():
    """Provide access to mock prediction results."""
    return MockAPIResponses.PREDICTIONS


# ========================================
# CLEANUP HELPERS
# ========================================

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically clean up temporary files after tests."""
    temp_files = []
    yield

    # Clean up any created temp files
    for filepath in temp_files:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass  # Ignore cleanup errors