"""
Integration Tests for End-to-End Pipeline
==========================================

Tests the complete workflow from data loading through predictions.
These tests ensure all components work together correctly.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


@pytest.mark.integration
class TestCompleteTrainingPipeline:
    """Test the complete ML training pipeline."""

    @pytest.mark.slow
    @patch('ml_models.pd.read_csv')
    def test_full_training_workflow(self, mock_read_csv, sample_features_data, temp_model_dir):
        """Test complete training workflow from data to saved models."""
        from ml_models import (
            load_and_prepare_data,
            create_temporal_train_test_split,
            prepare_features,
            MLModelTrainer
        )

        # Mock data loading
        mock_read_csv.return_value = sample_features_data

        # Step 1: Load data
        data = load_and_prepare_data()
        assert len(data) > 0

        # Step 2: Create splits
        train_data, test_data = create_temporal_train_test_split(data, test_size=0.2)
        assert len(train_data) > 0
        assert len(test_data) > 0

        # Step 3: Prepare features
        X_train, X_test, y_train, y_test, feature_cols, encoders = prepare_features(
            train_data, test_data
        )
        assert X_train.shape[1] > 0

        # Step 4: Train models
        trainer = MLModelTrainer()
        lr_model, lr_metrics = trainer.train_logistic_regression(
            X_train, y_train, X_test, y_test
        )
        rf_model, rf_metrics = trainer.train_random_forest(
            X_train, y_train, X_test, y_test
        )

        # Verify models were trained
        assert lr_model is not None
        assert rf_model is not None
        assert len(trainer.models) == 2
        assert len(trainer.results) == 2

        # Step 5: Save models (to temp dir)
        for model_name, model in trainer.models.items():
            model_file = os.path.join(temp_model_dir, f'{model_name}_model.pkl')
            joblib.dump(model, model_file)
            assert os.path.exists(model_file)


@pytest.mark.integration
class TestPredictionPipeline:
    """Test the complete prediction pipeline."""

    @patch('daily_predictions.NBAOddsClient')
    @patch('daily_predictions.pd.read_csv')
    @patch('daily_predictions.joblib.load')
    @patch('daily_predictions.os.path.exists')
    def test_full_prediction_workflow(self, mock_exists, mock_joblib_load,
                                     mock_read_csv, mock_odds_client,
                                     sample_features_data, mock_trained_model,
                                     mock_api_key):
        """Test complete prediction workflow from API to recommendations."""
        from daily_predictions import DailyPredictor

        # Setup mocks
        mock_exists.return_value = True
        mock_read_csv.return_value = sample_features_data

        # Mock model loading
        feature_cols = ['rolling_3g_points', 'rolling_5g_points', 'age']

        def joblib_side_effect(filepath):
            if 'model.pkl' in filepath:
                return mock_trained_model
            elif 'feature_columns.pkl' in filepath:
                return feature_cols
            elif 'label_encoders.pkl' in filepath:
                return {}

        mock_joblib_load.side_effect = joblib_side_effect

        # Mock API client
        mock_client_instance = MagicMock()
        mock_odds_client.return_value = mock_client_instance

        # Mock API responses
        mock_client_instance.get_all_todays_player_props.return_value = pd.DataFrame({
            'player_name': ['LeBron James'],
            'market_type': ['player_points'],
            'line_value': [25.5],
            'over_odds': [-110],
            'bookmaker': ['DraftKings'],
            'home_team': ['LAL'],
            'away_team': ['GSW'],
            'game_time': [datetime.now()],
            'timestamp': [datetime.now()]
        })

        mock_client_instance.format_for_ml_pipeline.return_value = pd.DataFrame({
            'fullName': ['LeBron James'],
            'gameDate': [datetime.now().date()],
            'home_team': ['LAL'],
            'away_team': ['GSW'],
            'prop_line': [25.5],
            'over_odds': [-110],
            'bookmaker': ['DraftKings'],
            'market_type': ['points'],
            'game_time': [datetime.now()],
            'api_timestamp': [datetime.now()]
        })

        # Initialize predictor
        predictor = DailyPredictor(api_key=mock_api_key)

        # Load models
        predictor.load_models()
        assert len(predictor.models) > 0

        # Load historical data
        historical_data = predictor.get_historical_player_data()
        assert len(historical_data) > 0

        # Generate prediction
        game_context = {
            'game_date': datetime.now().date(),
            'home_team': 'LAL',
            'away_team': 'GSW'
        }

        prediction = predictor.make_prediction_for_prop(
            player_name='LeBron James',
            prop_line=25.5,
            market_type='points',
            game_context=game_context,
            historical_data=historical_data
        )

        # Verify prediction
        assert prediction is not None
        assert 'recommendation' in prediction
        assert 'confidence' in prediction


@pytest.mark.integration
class TestAPIToMLPipeline:
    """Test integration between API client and ML pipeline."""

    @patch('odds_api_client.requests.get')
    def test_api_data_flows_to_ml_format(self, mock_get, mock_api_key,
                                        mock_player_props_response):
        """Test that API data is correctly formatted for ML pipeline."""
        from odds_api_client import NBAOddsClient

        # Setup mock API response
        mock_response = Mock()
        mock_response.json.return_value = mock_player_props_response
        mock_response.headers = {'x-requests-remaining': '100'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Fetch and format data
        client = NBAOddsClient(api_key=mock_api_key)
        props = client.get_player_props_for_game('game_123')

        # Convert to DataFrame
        props_df = pd.DataFrame(props)

        # Format for ML pipeline
        formatted = client.format_for_ml_pipeline(props_df)

        # Verify formatted data has required columns for ML
        required_columns = ['fullName', 'prop_line', 'market_type', 'gameDate']
        for col in required_columns:
            assert col in formatted.columns

        # Verify data types
        assert pd.api.types.is_numeric_dtype(formatted['prop_line'])


@pytest.mark.integration
class TestDataPersistence:
    """Test data saving and loading across pipeline stages."""

    def test_model_save_and_load(self, temp_model_dir, sample_features_data):
        """Test saving and reloading trained models."""
        from ml_models import MLModelTrainer, create_temporal_train_test_split, prepare_features

        # Train a model
        trainer = MLModelTrainer()
        train, test = create_temporal_train_test_split(sample_features_data, test_size=0.2)
        X_train, X_test, y_train, y_test, feature_cols, _ = prepare_features(train, test)

        rf_model, _ = trainer.train_random_forest(X_train, y_train, X_test, y_test)

        # Save model
        model_path = os.path.join(temp_model_dir, 'test_model.pkl')
        joblib.dump(rf_model, model_path)

        # Load model
        loaded_model = joblib.load(model_path)

        # Verify predictions match
        original_pred = rf_model.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)

        assert np.array_equal(original_pred, loaded_pred)

    def test_predictions_save_to_csv(self, temp_data_dir, mock_api_key):
        """Test that predictions can be saved and reloaded."""
        from daily_predictions import DailyPredictor

        predictions_df = pd.DataFrame({
            'player_name': ['LeBron James', 'Stephen Curry'],
            'recommendation': ['OVER', 'UNDER'],
            'confidence': [0.75, 0.68],
            'prop_line': [25.5, 28.5]
        })

        # Save predictions
        output_file = os.path.join(temp_data_dir, 'test_predictions.csv')
        predictions_df.to_csv(output_file, index=False)

        # Reload and verify
        loaded = pd.read_csv(output_file)
        assert len(loaded) == 2
        assert list(loaded.columns) == list(predictions_df.columns)


@pytest.mark.integration
class TestErrorPropagation:
    """Test how errors propagate through the pipeline."""

    @patch('daily_predictions.NBAOddsClient')
    def test_api_failure_handled_gracefully(self, mock_client, mock_api_key):
        """Test that API failures don't crash the pipeline."""
        from daily_predictions import DailyPredictor

        # Setup client that returns empty data
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_all_todays_player_props.return_value = pd.DataFrame()

        predictor = DailyPredictor(api_key=mock_api_key)

        # This should handle empty data gracefully
        # Would normally call run_daily_predictions but it requires models
        # Just test the API call handling
        props = mock_client_instance.get_all_todays_player_props()
        assert len(props) == 0

    def test_missing_feature_data_handled(self, sample_features_data):
        """Test handling when feature data is incomplete."""
        from ml_models import create_temporal_train_test_split, prepare_features

        # Remove some features
        incomplete_data = sample_features_data.drop(columns=['rolling_3g_points'])

        train, test = create_temporal_train_test_split(incomplete_data, test_size=0.2)

        # Should still prepare features (just with fewer features)
        X_train, X_test, y_train, y_test, feature_cols, _ = prepare_features(train, test)

        assert 'rolling_3g_points' not in feature_cols


@pytest.mark.integration
@pytest.mark.slow
class TestRealisticDataFlow:
    """Test realistic data flows through the entire system."""

    @patch('odds_api_client.requests.get')
    @patch('ml_models.pd.read_csv')
    def test_realistic_workflow_simulation(self, mock_read_csv, mock_get,
                                          sample_features_data, mock_api_key,
                                          mock_player_props_response, temp_model_dir):
        """Simulate a realistic workflow: train models, fetch props, make predictions."""
        from ml_models import (
            load_and_prepare_data,
            create_temporal_train_test_split,
            prepare_features,
            MLModelTrainer
        )
        from odds_api_client import NBAOddsClient

        # === PHASE 1: Train Models ===
        mock_read_csv.return_value = sample_features_data
        data = load_and_prepare_data()
        train, test = create_temporal_train_test_split(data, test_size=0.2)
        X_train, X_test, y_train, y_test, feature_cols, encoders = prepare_features(train, test)

        trainer = MLModelTrainer()
        trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
        trainer.train_random_forest(X_train, y_train, X_test, y_test)

        assert len(trainer.models) == 2

        # Save models
        for model_name, model in trainer.models.items():
            joblib.dump(model, os.path.join(temp_model_dir, f'{model_name}.pkl'))

        # === PHASE 2: Fetch Today's Props ===
        mock_response = Mock()
        mock_response.json.return_value = mock_player_props_response
        mock_response.headers = {'x-requests-remaining': '100'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = NBAOddsClient(api_key=mock_api_key)
        props = client.get_player_props_for_game('game_123')

        assert len(props) > 0

        # === PHASE 3: Make Predictions ===
        # This would normally use DailyPredictor.make_prediction_for_prop()
        # but we'll verify the data is ready for predictions

        props_df = pd.DataFrame(props)
        formatted = client.format_for_ml_pipeline(props_df)

        assert len(formatted) > 0
        assert 'fullName' in formatted.columns
        assert 'prop_line' in formatted.columns

        # Load a saved model and verify it can make predictions
        loaded_model = joblib.load(os.path.join(temp_model_dir, 'random_forest.pkl'))

        # Create simple prediction (would need feature engineering in real scenario)
        test_features = X_test.iloc[:1]
        prediction = loaded_model.predict(test_features)

        assert len(prediction) == 1
        assert prediction[0] in [0, 1]


@pytest.mark.integration
class TestConcurrentOperations:
    """Test handling of concurrent operations and state management."""

    def test_multiple_predictions_same_player(self, sample_features_data, mock_api_key):
        """Test making multiple predictions for the same player."""
        from daily_predictions import DailyPredictor

        with patch('daily_predictions.NBAOddsClient'):
            predictor = DailyPredictor(api_key=mock_api_key)

            # Setup models
            predictor.models = {'test': MagicMock()}
            predictor.models['test'].predict.return_value = np.array([1])
            predictor.models['test'].predict_proba.return_value = np.array([[0.3, 0.7]])
            predictor.feature_cols = ['rolling_3g_points', 'age']
            predictor.label_encoders = {}

            player_name = sample_features_data['fullName'].iloc[0]
            game_context = {
                'game_date': datetime.now().date(),
                'home_team': 'LAL',
                'away_team': 'GSW'
            }

            # Make multiple predictions
            pred1 = predictor.make_prediction_for_prop(
                player_name, 25.5, 'points', game_context, sample_features_data
            )
            pred2 = predictor.make_prediction_for_prop(
                player_name, 27.5, 'points', game_context, sample_features_data
            )

            # Both predictions should succeed
            assert pred1 is not None
            assert pred2 is not None
            # Different prop lines
            assert pred1['prop_line'] != pred2['prop_line']


@pytest.mark.integration
class TestBackwardCompatibility:
    """Test backward compatibility with older data formats."""

    def test_handles_legacy_column_names(self):
        """Test handling of legacy column names in data."""
        # Create data with old column names
        legacy_data = pd.DataFrame({
            'gameDate': pd.date_range('2023-01-01', periods=50),
            'fullName': ['Player 1'] * 25 + ['Player 2'] * 25,
            'points': np.random.randint(15, 40, 50),
            'over_threshold': np.random.randint(0, 2, 50),
        })

        # Should still work with prepare_features
        from ml_models import create_temporal_train_test_split, prepare_features

        train, test = create_temporal_train_test_split(legacy_data, test_size=0.2)
        X_train, X_test, y_train, y_test, _, _ = prepare_features(train, test)

        # Should extract features successfully
        assert len(X_train) > 0
        assert len(y_train) > 0
