"""
Pytest configuration and fixtures for NBA Betting Model tests
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
import tempfile
import os
from datetime import datetime, timedelta

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_nba_data():
    """Generate sample NBA data for testing"""
    dates = pd.date_range('2024-01-01', periods=100)
    np.random.seed(42)

    return pd.DataFrame({
        'date': dates,
        'player_id': np.random.randint(1, 50, 100),
        'player_name': [f'Player_{i}' for i in np.random.randint(1, 50, 100)],
        'team': [f'Team_{i}' for i in np.random.randint(1, 30, 100)],
        'opponent': [f'Team_{i}' for i in np.random.randint(1, 30, 100)],
        'home_game': np.random.choice([True, False], 100),
        'points': np.random.normal(20, 5, 100).clip(0, 50),
        'rebounds': np.random.normal(8, 2, 100).clip(0, 20),
        'assists': np.random.normal(5, 2, 100).clip(0, 15),
        'minutes': np.random.normal(30, 5, 100).clip(10, 48),
        'efficiency': np.random.normal(25, 5, 100).clip(0, 50),
        'prop_line': np.random.uniform(15, 25, 100),
        'odds_over': np.random.uniform(1.8, 2.2, 100),
        'odds_under': np.random.uniform(1.8, 2.2, 100)
    })


@pytest.fixture
def sample_model_data(sample_nba_data):
    """Generate data ready for model training"""
    data = sample_nba_data.copy()

    # Create target variable
    data['target'] = (data['points'] > data['prop_line']).astype(int)

    # Create features
    data['points_per_minute'] = data['points'] / data['minutes']
    data['efficiency_rating'] = data['efficiency'] / data['minutes']
    data['total_impact'] = data['points'] + data['rebounds'] + data['assists']

    # Add rolling features (simplified)
    data = data.sort_values(['player_id', 'date'])
    data['points_3avg'] = data.groupby('player_id')['points'].rolling(3).mean().reset_index(0, drop=True)
    data['rebounds_3avg'] = data.groupby('player_id')['rebounds'].rolling(3).mean().reset_index(0, drop=True)

    return data.dropna()


@pytest.fixture
def mock_model():
    """Create a mock trained model"""
    model = Mock()

    # Mock predictions
    model.predict.return_value = np.random.choice([0, 1], 50)
    model.predict_proba.return_value = np.random.dirichlet([1, 1], 50)

    # Mock feature importances
    model.feature_importances_ = np.random.rand(10)

    return model


@pytest.fixture
def mock_odds_api_response():
    """Mock API response for odds data"""
    return {
        'success': True,
        'data': [
            {
                'player_id': 1,
                'player_name': 'LeBron James',
                'team': 'Lakers',
                'opponent': 'Celtics',
                'prop_type': 'points',
                'line': 25.5,
                'over_odds': 1.90,
                'under_odds': 1.90,
                'date': '2024-01-15'
            },
            {
                'player_id': 2,
                'player_name': 'Kevin Durant',
                'team': 'Suns',
                'opponent': 'Warriors',
                'prop_type': 'points',
                'line': 28.5,
                'over_odds': 1.85,
                'under_odds': 1.95,
                'date': '2024-01-15'
            },
            {
                'player_id': 3,
                'player_name': 'Stephen Curry',
                'team': 'Warriors',
                'opponent': 'Suns',
                'prop_type': 'points',
                'line': 24.5,
                'over_odds': 1.95,
                'under_odds': 1.85,
                'date': '2024-01-15'
            }
        ]
    }


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_csv_data(sample_nba_data, temp_data_dir):
    """Create sample CSV files for testing"""
    csv_path = os.path.join(temp_data_dir, 'nba_data.csv')
    sample_nba_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_api_client():
    """Create mock API client"""
    client = Mock()
    client.get_odds.return_value = mock_odds_api_response()
    client.validate_api_key.return_value = True
    client.check_rate_limit.return_value = True
    return client


@pytest.fixture
def validation_data():
    """Create data specifically for validation testing"""
    # Create data with intentional issues
    dates = pd.date_range('2024-01-01', periods=200)
    np.random.seed(123)

    data = pd.DataFrame({
        'date': dates,
        'player_id': np.random.randint(1, 30, 200),
        'points': np.random.normal(20, 5, 200),
        'rebounds': np.random.normal(8, 2, 200),
        'assists': np.random.normal(5, 2, 200),
        'minutes': np.random.normal(30, 5, 200),
        'prop_line': np.random.uniform(18, 22, 200),
        # Add potential leakage features
        'over_threshold': np.random.choice([0, 1], 200),
        'current_game_points': np.random.normal(20, 5, 200),
        'target_score': np.random.normal(20, 5, 200)
    })

    # Create target
    data['target'] = (data['points'] > data['prop_line']).astype(int)

    return data


@pytest.fixture
def performance_test_data():
    """Create large dataset for performance testing"""
    dates = pd.date_range('2024-01-01', periods=10000)
    np.random.seed(456)

    return pd.DataFrame({
        'date': dates,
        'player_id': np.random.randint(1, 500, 10000),
        'points': np.random.normal(20, 5, 10000),
        'rebounds': np.random.normal(8, 2, 10000),
        'assists': np.random.normal(5, 2, 10000),
        'minutes': np.random.normal(30, 5, 10000),
        'efficiency': np.random.normal(25, 5, 10000),
        'prop_line': np.random.uniform(18, 22, 10000)
    })


@pytest.fixture
def corrupted_data():
    """Create data with quality issues for testing error handling"""
    dates = pd.date_range('2024-01-01', periods=50)

    data = pd.DataFrame({
        'date': dates,
        'player_id': [1, 2, 3, None, 5] * 10,  # Missing player_id
        'points': [20, np.inf, -np.inf, 25, None] * 10,  # Infinite and null values
        'rebounds': [8, None, 10, None, 7] * 10,
        'assists': [5, 3, None, 4, 2] * 10,
        'minutes': [35, None, 0, 40, 30] * 10,
        'prop_line': [20.5, 19.5, None, 21.5, 18.5] * 10
    })

    return data


@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings"""
    return {
        'random_state': 42,
        'test_size': 0.2,
        'cv_folds': 5,
        'confidence_threshold': 0.7,
        'max_features': 50,
        'min_samples_leaf': 5,
        'timeout_seconds': 30,
        'memory_limit_mb': 100
    }


# Custom pytest markers
def pytest_configure(config):
    """Configure custom markers"""
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
        "markers", "performance: marks tests as performance tests"
    )


# Pytest hooks
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup common test environment variables"""
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("ODDS_API_KEY", "test_key_12345")
    monkeypatch.setenv("PYTHONPATH", os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def mock_redis():
    """Mock Redis client for caching tests"""
    redis_mock = Mock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.exists.return_value = False
    return redis_mock


@pytest.fixture
def sample_ensemble_models():
    """Create sample ensemble of models"""
    models = []
    for i in range(3):
        model = Mock()
        model.predict.return_value = np.random.choice([0, 1], 50)
        model.predict_proba.return_value = np.random.dirichlet([1, 1], 50)
        model.feature_importances_ = np.random.rand(10)
        models.append(model)

    return models


# Helper functions for tests
def create_time_series_data(days=30, players=10):
    """Helper to create time series data"""
    dates = pd.date_range('2024-01-01', periods=days)
    data = []

    for player_id in range(1, players + 1):
        for date in dates:
            data.append({
                'date': date,
                'player_id': player_id,
                'points': np.random.normal(20, 5),
                'rebounds': np.random.normal(8, 2),
                'assists': np.random.normal(5, 2),
                'minutes': np.random.normal(30, 5)
            })

    return pd.DataFrame(data)


def assert_dataframe_valid(df, required_columns=None, min_rows=1):
    """Helper to validate DataFrame in tests"""
    assert isinstance(df, pd.DataFrame), "Expected pandas DataFrame"
    assert len(df) >= min_rows, f"DataFrame has fewer than {min_rows} rows"

    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        assert not missing_cols, f"Missing required columns: {missing_cols}"

    # Check for extreme null values
    null_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
    assert null_ratio < 0.5, "DataFrame has too many null values"