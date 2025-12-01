"""
Unit Tests for Daily Predictions Module
========================================

Tests the daily_predictions module with mocked dependencies.
Validates prediction logic without requiring actual models or API calls.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from final_predictions_optimized import OptimizedNBAPredictionsSystem as DailyPredictor


class TestDailyPredictorInitialization:
    """Test DailyPredictor initialization."""

    def test_init_with_api_key(self, mock_api_key):
        """Test initialization with explicit API key."""
        with patch('final_predictions_optimized.NBAOddsClient'):
            predictor = DailyPredictor(api_key=mock_api_key)

            assert predictor.api_key == mock_api_key
            assert predictor.models == {}
            assert predictor.feature_cols is None

    def test_init_with_env_var(self, mock_env_with_api_key):
        """Test initialization using environment variable."""
        with patch('final_predictions_optimized.NBAOddsClient'):
            predictor = DailyPredictor()

            assert predictor.api_key == mock_env_with_api_key

    def test_init_without_api_key(self, mock_env_without_api_key):
        """Test initialization fails without API key."""
        with pytest.raises(ValueError, match="API key required"):
            DailyPredictor()


class TestLoadModels:
    """Test model loading functionality."""

    @patch('daily_predictions.joblib.load')
    @patch('daily_predictions.os.path.exists')
    def test_load_models_success(self, mock_exists, mock_joblib_load, mock_api_key):
        """Test successful model loading."""
        mock_exists.return_value = True

        # Mock loaded objects
        mock_model = MagicMock()
        mock_features = ['feature1', 'feature2', 'feature3']
        mock_encoders = {'cat_feature': MagicMock()}

        def joblib_side_effect(filepath):
            if 'model.pkl' in filepath:
                return mock_model
            elif 'feature_columns.pkl' in filepath:
                return mock_features
            elif 'label_encoders.pkl' in filepath:
                return mock_encoders

        mock_joblib_load.side_effect = joblib_side_effect

        with patch('final_predictions_optimized.NBAOddsClient'):
            predictor = DailyPredictor(api_key=mock_api_key)
            predictor.load_models()

            assert len(predictor.models) > 0
            assert predictor.feature_cols == mock_features
            assert predictor.label_encoders == mock_encoders

    @patch('daily_predictions.joblib.load')
    @patch('daily_predictions.os.path.exists')
    def test_load_models_missing_features(self, mock_exists, mock_joblib_load, mock_api_key):
        """Test error when feature columns missing."""
        # Model exists but features don't
        def exists_side_effect(path):
            return 'feature_columns' not in path

        mock_exists.side_effect = exists_side_effect
        mock_joblib_load.return_value = MagicMock()

        with patch('final_predictions_optimized.NBAOddsClient'):
            predictor = DailyPredictor(api_key=mock_api_key)

            with pytest.raises(FileNotFoundError):
                predictor.load_models()

    @patch('daily_predictions.joblib.load')
    @patch('daily_predictions.os.path.exists')
    def test_load_models_no_models_found(self, mock_exists, mock_joblib_load, mock_api_key):
        """Test error when no models are found."""
        # Only feature columns exist
        def exists_side_effect(path):
            return 'feature_columns' in path

        mock_exists.side_effect = exists_side_effect
        mock_joblib_load.return_value = ['feature1', 'feature2']

        with patch('final_predictions_optimized.NBAOddsClient'):
            predictor = DailyPredictor(api_key=mock_api_key)

            with pytest.raises(ValueError, match="No models loaded"):
                predictor.load_models()


class TestHistoricalDataLoading:
    """Test historical player data loading."""

    @patch('daily_predictions.pd.read_csv')
    def test_get_historical_player_data_success(self, mock_read_csv, mock_api_key,
                                               sample_features_data):
        """Test successful historical data loading."""
        mock_read_csv.return_value = sample_features_data

        with patch('final_predictions_optimized.NBAOddsClient'):
            predictor = DailyPredictor(api_key=mock_api_key)
            data = predictor.get_historical_player_data()

            assert len(data) > 0
            assert 'gameDate' in data.columns
            assert 'fullName' in data.columns

    @patch('daily_predictions.pd.read_csv')
    def test_get_historical_player_data_file_not_found(self, mock_read_csv, mock_api_key):
        """Test error handling when data file missing."""
        mock_read_csv.side_effect = FileNotFoundError("File not found")

        with patch('final_predictions_optimized.NBAOddsClient'):
            predictor = DailyPredictor(api_key=mock_api_key)

            with pytest.raises(FileNotFoundError):
                predictor.get_historical_player_data()


class TestGenerateGameContextFeatures:
    """Test feature generation for predictions."""

    def test_generate_features_success(self, mock_api_key, sample_features_data):
        """Test successful feature generation."""
        with patch('final_predictions_optimized.NBAOddsClient'):
            predictor = DailyPredictor(api_key=mock_api_key)

            player_name = sample_features_data['fullName'].iloc[0]
            game_date = datetime.now().date()

            features = predictor.generate_game_context_features(
                player_name=player_name,
                game_date=game_date,
                home_team='LAL',
                away_team='GSW',
                historical_data=sample_features_data
            )

            assert features is not None
            assert 'fullName' in features.index

    def test_generate_features_unknown_player(self, mock_api_key, sample_features_data):
        """Test handling of unknown player."""
        with patch('final_predictions_optimized.NBAOddsClient'):
            predictor = DailyPredictor(api_key=mock_api_key)

            features = predictor.generate_game_context_features(
                player_name='Unknown Player',
                game_date=datetime.now().date(),
                home_team='LAL',
                away_team='GSW',
                historical_data=sample_features_data
            )

            assert features is None


class TestMakePrediction:
    """Test prediction generation."""

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_make_prediction_success(self, mock_client, mock_api_key,
                                    sample_features_data, mock_trained_model):
        """Test successful prediction generation."""
        predictor = DailyPredictor(api_key=mock_api_key)

        # Setup predictor with test data
        predictor.models = {
            'logistic_regression': mock_trained_model,
            'random_forest': mock_trained_model,
            'xgboost': mock_trained_model
        }
        predictor.feature_cols = ['rolling_3g_points', 'rolling_5g_points', 'age']
        predictor.label_encoders = {}

        player_name = sample_features_data['fullName'].iloc[0]
        game_context = {
            'game_date': datetime.now().date(),
            'home_team': 'LAL',
            'away_team': 'GSW'
        }

        prediction = predictor.make_prediction_for_prop(
            player_name=player_name,
            prop_line=25.5,
            market_type='points',
            game_context=game_context,
            historical_data=sample_features_data
        )

        assert prediction is not None
        assert 'player_name' in prediction
        assert 'recommendation' in prediction
        assert 'confidence' in prediction
        assert prediction['recommendation'] in ['OVER', 'UNDER']
        assert 0 <= prediction['confidence'] <= 1

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_make_prediction_unknown_player(self, mock_client, mock_api_key,
                                           sample_features_data):
        """Test prediction for unknown player."""
        predictor = DailyPredictor(api_key=mock_api_key)
        predictor.models = {'test_model': MagicMock()}
        predictor.feature_cols = ['rolling_3g_points']
        predictor.label_encoders = {}

        game_context = {
            'game_date': datetime.now().date(),
            'home_team': 'LAL',
            'away_team': 'GSW'
        }

        prediction = predictor.make_prediction_for_prop(
            player_name='Unknown Player',
            prop_line=25.5,
            market_type='points',
            game_context=game_context,
            historical_data=sample_features_data
        )

        assert prediction is None

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_make_prediction_model_disagreement(self, mock_client, mock_api_key,
                                               sample_features_data):
        """Test prediction when models disagree."""
        predictor = DailyPredictor(api_key=mock_api_key)

        # Create models with different predictions
        model_over = MagicMock()
        model_over.predict.return_value = np.array([1])
        model_over.predict_proba.return_value = np.array([[0.3, 0.7]])

        model_under = MagicMock()
        model_under.predict.return_value = np.array([0])
        model_under.predict_proba.return_value = np.array([[0.6, 0.4]])

        predictor.models = {
            'model1': model_over,
            'model2': model_under,
        }
        predictor.feature_cols = ['rolling_3g_points']
        predictor.label_encoders = {}

        player_name = sample_features_data['fullName'].iloc[0]
        game_context = {
            'game_date': datetime.now().date(),
            'home_team': 'LAL',
            'away_team': 'GSW'
        }

        prediction = predictor.make_prediction_for_prop(
            player_name=player_name,
            prop_line=25.5,
            market_type='points',
            game_context=game_context,
            historical_data=sample_features_data
        )

        # Should return prediction with split agreement
        assert prediction is not None
        assert prediction['models_agreement'] == '1/2'


class TestDisplayPredictions:
    """Test prediction display functionality."""

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_display_predictions_with_results(self, mock_client, mock_api_key, capfd):
        """Test displaying predictions with results."""
        predictor = DailyPredictor(api_key=mock_api_key)

        predictions_df = pd.DataFrame({
            'player_name': ['LeBron James'],
            'market_type': ['points'],
            'prop_line': [25.5],
            'recommendation': ['OVER'],
            'confidence': [0.75],
            'models_agreement': ['3/3'],
            'over_odds': [-110],
            'bookmaker': ['DraftKings'],
            'home_team': ['LAL'],
            'away_team': ['GSW']
        })

        predictor.display_predictions(predictions_df)

        captured = capfd.readouterr()
        assert 'LeBron James' in captured.out
        assert 'OVER' in captured.out

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_display_predictions_empty(self, mock_client, mock_api_key, capfd):
        """Test displaying empty predictions."""
        predictor = DailyPredictor(api_key=mock_api_key)
        predictor.display_predictions(pd.DataFrame())

        captured = capfd.readouterr()
        assert 'No high-confidence predictions' in captured.out


class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_prediction_with_nan_values(self, mock_client, mock_api_key):
        """Test handling of NaN values in features."""
        predictor = DailyPredictor(api_key=mock_api_key)

        # Create data with NaN values
        data = pd.DataFrame({
            'fullName': ['Test Player'],
            'gameDate': [datetime.now()],
            'rolling_3g_points': [np.nan],
            'age': [25]
        })

        predictor.models = {'test': MagicMock()}
        predictor.feature_cols = ['rolling_3g_points', 'age']
        predictor.label_encoders = {}

        game_context = {
            'game_date': datetime.now().date(),
            'home_team': 'LAL',
            'away_team': 'GSW'
        }

        # Should handle NaN values without crashing
        prediction = predictor.make_prediction_for_prop(
            player_name='Test Player',
            prop_line=25.5,
            market_type='points',
            game_context=game_context,
            historical_data=data
        )

        # Might be None or valid prediction depending on data
        assert prediction is None or isinstance(prediction, dict)

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_prediction_with_missing_features(self, mock_client, mock_api_key,
                                             sample_features_data):
        """Test prediction when required features are missing."""
        predictor = DailyPredictor(api_key=mock_api_key)

        predictor.models = {'test': MagicMock()}
        predictor.feature_cols = ['nonexistent_feature']
        predictor.label_encoders = {}

        player_name = sample_features_data['fullName'].iloc[0]
        game_context = {
            'game_date': datetime.now().date(),
            'home_team': 'LAL',
            'away_team': 'GSW'
        }

        # Should handle missing features gracefully
        try:
            prediction = predictor.make_prediction_for_prop(
                player_name=player_name,
                prop_line=25.5,
                market_type='points',
                game_context=game_context,
                historical_data=sample_features_data
            )
            # Should return None or handle error
            assert prediction is None or isinstance(prediction, dict)
        except Exception as e:
            # Some exceptions are acceptable
            assert isinstance(e, (KeyError, ValueError))


@pytest.mark.parametrize("confidence,agreement,should_include", [
    (0.75, 1.0, True),   # High confidence, full agreement
    (0.55, 0.0, True),   # Medium confidence, full disagreement
    (0.45, 0.5, False),  # Low confidence
    (0.65, 0.66, False), # Partial agreement
])
def test_high_confidence_filtering(confidence, agreement, should_include):
    """Test filtering logic for high-confidence predictions."""
    predictions_df = pd.DataFrame({
        'confidence': [confidence],
        'over_percentage': [agreement],
        'player_name': ['Test Player']
    })

    # Filter matching the code in daily_predictions.py
    high_confidence = predictions_df[
        (predictions_df['confidence'] > 0.6) &
        (predictions_df['over_percentage'].isin([0.0, 1.0]))
    ]

    assert (len(high_confidence) > 0) == should_include
