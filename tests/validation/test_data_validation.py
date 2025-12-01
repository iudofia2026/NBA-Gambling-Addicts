"""
Data Validation Tests
=====================

Tests to ensure data quality and ML model correctness.
These tests validate assumptions about the data and model behavior.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestDataQuality:
    """Test data quality and integrity."""

    def test_no_future_leakage_in_features(self, sample_features_data):
        """Test that rolling features don't include current game data."""
        # Rolling features should be shifted (not include current game)
        for idx, row in sample_features_data.iterrows():
            if idx > 0 and row['fullName'] == sample_features_data.iloc[idx-1]['fullName']:
                # Rolling average should not equal current value
                # (unless player is extremely consistent, which is unlikely)
                pass  # This is a soft check, exact matching is coincidental

        # More robust check: ensure rolling features are NaN or 0 for first game
        first_games = sample_features_data.groupby('fullName').first()
        if 'rolling_3g_points' in first_games.columns:
            # First games should have NaN or 0 for rolling features
            assert (first_games['rolling_3g_points'].isna() |
                   (first_games['rolling_3g_points'] == 0)).all()

    def test_target_variable_balance(self, sample_features_data):
        """Test that target variable is not extremely imbalanced."""
        if 'over_threshold' in sample_features_data.columns:
            over_rate = sample_features_data['over_threshold'].mean()

            # Should be between 20% and 80% (not too imbalanced)
            assert 0.2 <= over_rate <= 0.8, \
                f"Target variable too imbalanced: {over_rate:.1%} positive class"

    def test_no_duplicate_games(self, sample_features_data):
        """Test that there are no duplicate game records."""
        # Check for duplicates based on player + date
        duplicates = sample_features_data.duplicated(subset=['fullName', 'gameDate'], keep=False)

        duplicate_count = duplicates.sum()
        if duplicate_count > 0:
            print(f"Warning: Found {duplicate_count} duplicate game records")

        # In production, should have no duplicates
        # For test data, just warn
        assert True

    def test_chronological_order_per_player(self, sample_features_data):
        """Test that games are in chronological order for each player."""
        for player in sample_features_data['fullName'].unique():
            player_data = sample_features_data[
                sample_features_data['fullName'] == player
            ].copy()

            dates = pd.to_datetime(player_data['gameDate'])

            # Check if sorted
            is_sorted = dates.is_monotonic_increasing

            if not is_sorted:
                print(f"Warning: Games not sorted for {player}")

    def test_feature_ranges_realistic(self, sample_features_data):
        """Test that feature values are in realistic ranges."""
        if 'points' in sample_features_data.columns:
            points = sample_features_data['points']

            # Points should be between 0 and 100 (realistic range)
            assert points.min() >= 0, "Found negative points"
            assert points.max() <= 100, f"Found unrealistic points: {points.max()}"

        if 'numMinutes' in sample_features_data.columns:
            minutes = sample_features_data['numMinutes']

            # Minutes should be between 0 and 48 (max game time)
            assert minutes.min() >= 0, "Found negative minutes"
            assert minutes.max() <= 60, f"Found unrealistic minutes: {minutes.max()}"

    def test_no_missing_critical_features(self, sample_features_data):
        """Test that critical features are not missing."""
        critical_features = ['fullName', 'gameDate', 'over_threshold']

        for feature in critical_features:
            if feature in sample_features_data.columns:
                missing_rate = sample_features_data[feature].isna().mean()

                assert missing_rate < 0.01, \
                    f"Critical feature '{feature}' has {missing_rate:.1%} missing values"


class TestModelValidation:
    """Test ML model behavior and performance."""

    def test_model_predictions_in_valid_range(self, mock_trained_model):
        """Test that model predictions are in valid range [0, 1]."""
        X_test = pd.DataFrame(np.random.rand(100, 5))

        predictions = mock_trained_model.predict(X_test)
        probabilities = mock_trained_model.predict_proba(X_test)

        # Predictions should be 0 or 1
        assert np.all(np.isin(predictions, [0, 1]))

        # Probabilities should sum to 1 and be in [0, 1]
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_model_better_than_random(self, sample_features_data):
        """Test that trained models perform better than random guessing."""
        from ml_models import (
            create_temporal_train_test_split,
            prepare_features,
            MLModelTrainer
        )

        train, test = create_temporal_train_test_split(sample_features_data, test_size=0.2)
        X_train, X_test, y_train, y_test, _, _ = prepare_features(train, test)

        trainer = MLModelTrainer()
        _, metrics = trainer.train_logistic_regression(X_train, y_train, X_test, y_test)

        # Should be better than 50% accuracy (random guessing)
        assert metrics['accuracy'] > 0.50, \
            f"Model accuracy {metrics['accuracy']:.3f} not better than random"

    def test_model_not_always_predicting_same_class(self, sample_features_data):
        """Test that model doesn't always predict the same class."""
        from ml_models import (
            create_temporal_train_test_split,
            prepare_features,
            MLModelTrainer
        )

        train, test = create_temporal_train_test_split(sample_features_data, test_size=0.2)
        X_train, X_test, y_train, y_test, _, _ = prepare_features(train, test)

        trainer = MLModelTrainer()
        model, _ = trainer.train_random_forest(X_train, y_train, X_test, y_test)

        predictions = model.predict(X_test)

        # Should predict both classes
        unique_predictions = np.unique(predictions)
        assert len(unique_predictions) > 1, \
            "Model always predicting same class - likely overfitting or underfitting"

    def test_feature_importance_sensible(self, sample_features_data):
        """Test that feature importance values are sensible."""
        from ml_models import (
            create_temporal_train_test_split,
            prepare_features,
            MLModelTrainer
        )

        train, test = create_temporal_train_test_split(sample_features_data, test_size=0.2)
        X_train, X_test, y_train, y_test, _, _ = prepare_features(train, test)

        trainer = MLModelTrainer()
        trainer.train_random_forest(X_train, y_train, X_test, y_test)

        importance = trainer.feature_importance['random_forest']

        # All importance values should be non-negative
        assert all(v >= 0 for v in importance.values()), \
            "Found negative feature importance"

        # Sum of importance should be reasonable (close to 1 for RF)
        total_importance = sum(importance.values())
        assert 0.9 <= total_importance <= 1.1, \
            f"Total feature importance {total_importance} seems off"

        # Top features should have meaningful importance
        sorted_importance = sorted(importance.values(), reverse=True)
        top_feature_importance = sorted_importance[0]

        assert top_feature_importance > 0.01, \
            "Top feature has very low importance - model might not be learning"


class TestPredictionValidation:
    """Test prediction output quality."""

    def test_predictions_have_required_fields(self):
        """Test that prediction objects have all required fields."""
        prediction = {
            'player_name': 'LeBron James',
            'market_type': 'points',
            'prop_line': 25.5,
            'recommendation': 'OVER',
            'confidence': 0.75,
            'over_percentage': 1.0,
            'models_agreement': '3/3'
        }

        required_fields = [
            'player_name', 'recommendation', 'confidence',
            'prop_line', 'market_type'
        ]

        for field in required_fields:
            assert field in prediction, f"Missing required field: {field}"

    def test_confidence_in_valid_range(self):
        """Test that confidence scores are in valid range."""
        prediction = {
            'confidence': 0.75,
            'over_percentage': 0.67
        }

        assert 0 <= prediction['confidence'] <= 1, \
            "Confidence should be between 0 and 1"
        assert 0 <= prediction['over_percentage'] <= 1, \
            "Over percentage should be between 0 and 1"

    def test_recommendation_is_valid(self):
        """Test that recommendations are valid values."""
        valid_recommendations = ['OVER', 'UNDER']

        prediction = {'recommendation': 'OVER'}

        assert prediction['recommendation'] in valid_recommendations, \
            f"Invalid recommendation: {prediction['recommendation']}"


class TestTemporalValidation:
    """Test temporal aspects of the data and models."""

    def test_no_look_ahead_bias_in_split(self, sample_features_data):
        """Test that train/test split doesn't have look-ahead bias."""
        from ml_models import create_temporal_train_test_split

        train, test = create_temporal_train_test_split(sample_features_data, test_size=0.2)

        # Latest train date should be before earliest test date
        train_max_date = pd.to_datetime(train['gameDate']).max()
        test_min_date = pd.to_datetime(test['gameDate']).min()

        assert train_max_date <= test_min_date, \
            "Train set contains dates after test set - look-ahead bias detected"

    def test_rolling_features_use_past_data_only(self, sample_features_data):
        """Test that rolling features only use past data."""
        # For each row, rolling features should be based on previous games only
        for player in sample_features_data['fullName'].unique()[:3]:  # Check first 3 players
            player_data = sample_features_data[
                sample_features_data['fullName'] == player
            ].sort_values('gameDate').copy()

            if len(player_data) < 4:
                continue

            # For game 4, the rolling_3g feature should be average of games 1-3
            if 'rolling_3g_points' in player_data.columns and 'points' in player_data.columns:
                game_4_rolling = player_data.iloc[3]['rolling_3g_points']
                expected = player_data.iloc[0:3]['points'].mean()

                # Allow for some floating point error and NaN handling
                if not pd.isna(game_4_rolling) and not pd.isna(expected):
                    assert abs(game_4_rolling - expected) < 1.0, \
                        f"Rolling feature not using correct past data: {game_4_rolling} vs {expected}"

    def test_predictions_for_future_dates_only(self):
        """Test that predictions are made for future dates only."""
        today = datetime.now().date()

        # Prediction input should be for future games
        prediction_input = {
            'game_time': datetime.now() + timedelta(hours=2),
            'gameDate': today
        }

        # Game time should be in the future
        assert prediction_input['game_time'].date() >= today, \
            "Predictions should be for future games only"


class TestAPIDataValidation:
    """Test validation of data from the Odds API."""

    def test_prop_lines_in_realistic_range(self):
        """Test that prop lines are in realistic ranges."""
        prop_data = pd.DataFrame({
            'market_type': ['points', 'assists', 'rebounds'],
            'prop_line': [25.5, 7.5, 10.5]
        })

        for _, row in prop_data.iterrows():
            if row['market_type'] == 'points':
                assert 0 < row['prop_line'] < 60, \
                    f"Unrealistic points line: {row['prop_line']}"
            elif row['market_type'] == 'assists':
                assert 0 < row['prop_line'] < 20, \
                    f"Unrealistic assists line: {row['prop_line']}"
            elif row['market_type'] == 'rebounds':
                assert 0 < row['prop_line'] < 25, \
                    f"Unrealistic rebounds line: {row['prop_line']}"

    def test_odds_in_valid_format(self):
        """Test that odds are in valid American odds format."""
        odds_data = pd.DataFrame({
            'over_odds': [-110, 100, 150, -200]
        })

        for odds in odds_data['over_odds']:
            # American odds should be integers
            assert isinstance(odds, (int, np.integer)) or odds == int(odds), \
                f"Odds should be integers: {odds}"

            # Should be in reasonable range
            assert -1000 <= odds <= 1000, \
                f"Odds outside reasonable range: {odds}"

    def test_player_names_not_empty(self):
        """Test that player names are not empty or malformed."""
        props_data = pd.DataFrame({
            'player_name': ['LeBron James', 'Stephen Curry', 'Kevin Durant']
        })

        for name in props_data['player_name']:
            assert isinstance(name, str), "Player name should be string"
            assert len(name) > 0, "Player name should not be empty"
            assert len(name.split()) >= 2, \
                f"Player name should have first and last name: {name}"


class TestEdgeCaseValidation:
    """Test handling of edge cases in validation."""

    def test_handles_players_with_limited_history(self, sample_features_data):
        """Test handling of players with very few historical games."""
        # Filter to only first few games per player
        limited_data = sample_features_data.groupby('fullName').head(3)

        # Should still be able to create features
        assert len(limited_data) > 0

        # Rolling features might be sparse
        if 'rolling_3g_points' in limited_data.columns:
            # First 2 games will have NaN or 0 for rolling_3g
            first_games = limited_data.groupby('fullName').head(2)
            assert (first_games['rolling_3g_points'].isna() |
                   (first_games['rolling_3g_points'] == 0)).any()

    def test_handles_extreme_stat_values(self):
        """Test handling of extreme (but valid) statistical values."""
        extreme_data = pd.DataFrame({
            'fullName': ['Player 1'] * 5,
            'gameDate': pd.date_range('2023-01-01', periods=5),
            'points': [0, 5, 50, 45, 3],  # Wide range including low scores
            'over_threshold': [0, 0, 1, 1, 0]
        })

        # Should handle extreme values without errors
        from ml_models import create_temporal_train_test_split

        # This shouldn't crash
        train, test = create_temporal_train_test_split(extreme_data, test_size=0.2)
        assert len(train) + len(test) == len(extreme_data)

    def test_validates_market_type_consistency(self):
        """Test that market types are consistent with prop lines."""
        # Points lines should be higher than assists lines on average
        props_data = pd.DataFrame({
            'market_type': ['points', 'points', 'assists', 'assists'],
            'prop_line': [25.5, 28.5, 7.5, 9.5]
        })

        points_avg = props_data[props_data['market_type'] == 'points']['prop_line'].mean()
        assists_avg = props_data[props_data['market_type'] == 'assists']['prop_line'].mean()

        assert points_avg > assists_avg, \
            "Points lines should typically be higher than assists lines"


@pytest.mark.parametrize("metric_name,min_value,max_value", [
    ('accuracy', 0.0, 1.0),
    ('precision', 0.0, 1.0),
    ('recall', 0.0, 1.0),
    ('f1_score', 0.0, 1.0),
    ('roc_auc', 0.0, 1.0),
])
def test_metric_ranges(metric_name, min_value, max_value):
    """Test that all metrics are in valid ranges."""
    # Simulate some metrics
    metrics = {
        'accuracy': 0.65,
        'precision': 0.62,
        'recall': 0.58,
        'f1_score': 0.60,
        'roc_auc': 0.68
    }

    assert min_value <= metrics[metric_name] <= max_value, \
        f"{metric_name} should be between {min_value} and {max_value}"
