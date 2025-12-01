"""
Integration Tests for Optimized NBA Predictions Pipeline
======================================================

End-to-end tests that verify the complete pipeline works together.
Tests integration between models, API client, data processing, and predictions.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import joblib

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from final_predictions_optimized import OptimizedNBAPredictionsSystem


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline functionality."""

    @pytest.fixture
    def full_test_setup(self, test_data_dir, sample_features_data, mock_api_key):
        """Create complete test setup with models and data."""
        # Create model directory
        model_dir = os.path.join(test_data_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)

        # Create data directory
        data_dir = os.path.join(test_data_dir, 'data', 'processed')
        os.makedirs(data_dir, exist_ok=True)

        # Save engineered features data
        features_file = os.path.join(data_dir, 'engineered_features.csv')
        sample_features_data.to_csv(features_file, index=False)

        # Create mock models
        rf_model = MagicMock()
        rf_model.predict.return_value = np.array([1, 0, 1, 0])
        rf_model.predict_proba.return_value = np.array([
            [0.3, 0.7],
            [0.6, 0.4],
            [0.2, 0.8],
            [0.55, 0.45]
        ])

        xgb_model = MagicMock()
        xgb_model.predict.return_value = np.array([1, 0, 1, 0])
        xgb_model.predict_proba.return_value = np.array([
            [0.25, 0.75],
            [0.65, 0.35],
            [0.15, 0.85],
            [0.6, 0.4]
        ])

        # Save models
        joblib.dump(rf_model, os.path.join(model_dir, 'random_forest_model.pkl'))
        joblib.dump(xgb_model, os.path.join(model_dir, 'xgboost_model.pkl'))

        # Update sample data with team names
        sample_features_data['playerteamName'] = np.random.choice(['LAL', 'GSW', 'PHX'], len(sample_features_data))
        sample_features_data['opponentteamName'] = np.random.choice(['GSW', 'LAL', 'PHX'], len(sample_features_data))
        sample_features_data.to_csv(features_file, index=False)

        return {
            'model_dir': model_dir,
            'data_dir': data_dir,
            'features_file': features_file,
            'sample_data': sample_features_data
        }

    @patch('final_predictions_optimized.NBAOddsClient')
    @patch('final_predictions_optimized.pd.DataFrame.to_csv')
    def test_complete_pipeline_success(self, mock_save, mock_client_class, full_test_setup,
                                      mock_api_key, mock_env_with_api_key):
        """Test complete pipeline from initialization to results."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create sample props data
        props_data = pd.DataFrame({
            'fullName': ['LeBron James', 'Stephen Curry', 'Kevin Durant'],
            'market_type': ['points', 'points', 'rebounds'],
            'prop_line': [25.5, 28.5, 10.5],
            'over_odds': [-110, -115, -105],
            'bookmaker': ['DraftKings', 'FanDuel', 'DraftKings'],
            'home_team': ['Los Angeles Lakers', 'Golden State Warriors', 'Phoenix Suns'],
            'away_team': ['Golden State Warriors', 'Phoenix Suns', 'Los Angeles Lakers'],
            'game_time': [datetime.now()] * 3,
            'gameDate': [datetime.now().date()] * 3,
            'playerteamName': ['LAL', 'GSW', 'PHX']
        })

        mock_client.get_all_todays_player_props.return_value = props_data
        mock_client.format_for_ml_pipeline.return_value = props_data

        # Mock working directory change
        original_cwd = os.getcwd()
        try:
            os.chdir(full_test_setup['model_dir'].replace('/models', ''))

            # Run complete pipeline
            system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
            predictions = system.run_optimized_predictions()

            # Verify results
            assert predictions is not None
            assert isinstance(predictions, list)

            if predictions:  # If any predictions were generated
                for pred in predictions:
                    assert 'player_name' in pred
                    assert 'market_type' in pred
                    assert 'recommendation' in pred
                    assert 'confidence' in pred
                    assert 'optimized_insights' in pred

                    # Verify data integrity
                    assert pred['recommendation'] in ['OVER', 'UNDER']
                    assert 0 <= pred['confidence'] <= 1

        finally:
            os.chdir(original_cwd)

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_pipeline_model_loading_failure(self, mock_client_class, mock_api_key):
        """Test pipeline behavior when model loading fails."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)

        with patch.object(system, 'load_models', return_value=False):
            result = system.run_optimized_predictions()

            assert result is None

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_pipeline_no_props_available(self, mock_client_class, full_test_setup,
                                        mock_api_key, mock_env_with_api_key):
        """Test pipeline when no player props are available."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_all_todays_player_props.return_value = pd.DataFrame()

        original_cwd = os.getcwd()
        try:
            os.chdir(full_test_setup['model_dir'].replace('/models', ''))

            system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
            result = system.run_optimized_predictions()

            assert result is None

        finally:
            os.chdir(original_cwd)

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_pipeline_api_failure(self, mock_client_class, full_test_setup,
                                 mock_api_key, mock_env_with_api_key):
        """Test pipeline behavior when API fails."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_all_todays_player_props.side_effect = Exception("API Error")

        original_cwd = os.getcwd()
        try:
            os.chdir(full_test_setup['model_dir'].replace('/models', ''))

            system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
            result = system.run_optimized_predictions()

            assert result is None

        finally:
            os.chdir(original_cwd)


class TestModelDataIntegration:
    """Test integration between models and data processing."""

    @pytest.fixture
    def model_data_setup(self, full_test_setup, mock_api_key):
        """Setup for testing model-data integration."""
        # Create system and load models
        original_cwd = os.getcwd()
        try:
            os.chdir(full_test_setup['model_dir'].replace('/models', ''))

            system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
            system.load_models()

            yield system, full_test_setup['sample_data']

        finally:
            os.chdir(original_cwd)

    def test_model_loading_with_data(self, model_data_setup):
        """Test that models load correctly with data."""
        system, data = model_data_setup

        assert len(system.models) > 0
        assert system.historical_data is not None
        assert len(system.historical_data) > 0

    def test_prediction_features_integration(self, model_data_setup):
        """Test that prediction features integrate properly with loaded models."""
        system, data = model_data_setup

        player_name = data['fullName'].iloc[0]
        game_context = {
            'player_team': data['playerteamName'].iloc[0],
            'opponent_team': data['opponentteamName'].iloc[0]
        }
        baseline_points = data['points'].tail(10).mean()

        result = system.calculate_optimized_prediction(
            player_name, game_context, baseline_points
        )

        assert 'predicted_points' in result
        assert 'confidence_score' in result
        assert 'feature_breakdown' in result

        # Verify feature breakdown structure
        features = result['feature_breakdown']
        assert 'form_factor' in features
        assert 'matchup_factor' in features
        assert 'chemistry_factor' in features


class TestDataFlowIntegration:
    """Test data flow through the pipeline."""

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_api_data_to_prediction_flow(self, mock_client_class, full_test_setup,
                                        mock_api_key, mock_env_with_api_key):
        """Test data flow from API to predictions."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create realistic API data
        api_data = pd.DataFrame({
            'player_name': ['LeBron James'],
            'market_type': ['player_points'],
            'line_value': [25.5],
            'over_odds': [-110],
            'bookmaker': ['DraftKings'],
            'home_team': ['Los Angeles Lakers'],
            'away_team': ['Golden State Warriors'],
            'game_time': [datetime.now()],
            'timestamp': [datetime.now()]
        })

        mock_client.get_all_todays_player_props.return_value = api_data

        # Test formatting
        formatted_data = mock_client.format_for_ml_pipeline(api_data)
        assert len(formatted_data) == 1
        assert 'fullName' in formatted_data.columns
        assert formatted_data['market_type'].iloc[0] == 'points'

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_historical_data_integration(self, mock_client_class, full_test_setup,
                                        mock_api_key, mock_env_with_api_key):
        """Test historical data integration with predictions."""
        original_cwd = os.getcwd()
        try:
            os.chdir(full_test_setup['model_dir'].replace('/models', ''))

            system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
            system.load_models()

            # Test that historical data is properly loaded and accessible
            assert system.historical_data is not None
            assert 'fullName' in system.historical_data.columns
            assert 'points' in system.historical_data.columns
            assert 'gameDate' in system.historical_data.columns

            # Test that data can be used for analysis
            player_name = system.historical_data['fullName'].iloc[0]
            form_result = system.analyze_player_form(player_name)

            assert 'form_confidence' in form_result

        finally:
            os.chdir(original_cwd)


class TestPerformanceIntegration:
    """Test performance and scalability integration."""

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_prediction_performance(self, mock_client_class, full_test_setup,
                                   mock_api_key, mock_env_with_api_key):
        """Test that predictions are generated within reasonable time."""
        import time

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create larger dataset for performance testing
        large_props_data = pd.DataFrame({
            'fullName': [f'Player_{i}' for i in range(50)],
            'market_type': ['points'] * 50,
            'prop_line': [25.5] * 50,
            'over_odds': [-110] * 50,
            'bookmaker': ['DraftKings'] * 50,
            'home_team': ['Team_A'] * 25 + ['Team_B'] * 25,
            'away_team': ['Team_B'] * 25 + ['Team_A'] * 25,
            'game_time': [datetime.now()] * 50,
            'gameDate': [datetime.now().date()] * 50
        })

        mock_client.get_all_todays_player_props.return_value = large_props_data
        mock_client.format_for_ml_pipeline.return_value = large_props_data

        original_cwd = os.getcwd()
        try:
            os.chdir(full_test_setup['model_dir'].replace('/models', ''))

            system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
            system.load_models()

            # Measure prediction time
            start_time = time.time()
            predictions = system.run_optimized_predictions()
            end_time = time.time()

            prediction_time = end_time - start_time

            # Should complete within reasonable time (adjust threshold as needed)
            assert prediction_time < 30.0  # 30 seconds max for 50 predictions

        finally:
            os.chdir(original_cwd)

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_memory_usage_integration(self, mock_client_class, full_test_setup,
                                    mock_api_key, mock_env_with_api_key):
        """Test that memory usage stays reasonable during pipeline execution."""
        import psutil
        import os

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create moderate dataset
        props_data = pd.DataFrame({
            'fullName': [f'Player_{i}' for i in range(20)],
            'market_type': ['points'] * 20,
            'prop_line': [25.5] * 20,
            'over_odds': [-110] * 20,
            'bookmaker': ['DraftKings'] * 20,
            'home_team': ['Team_A'] * 10 + ['Team_B'] * 10,
            'away_team': ['Team_B'] * 10 + ['Team_A'] * 10,
            'game_time': [datetime.now()] * 20,
            'gameDate': [datetime.now().date()] * 20
        })

        mock_client.get_all_todays_player_props.return_value = props_data
        mock_client.format_for_ml_pipeline.return_value = props_data

        original_cwd = os.getcwd()
        try:
            os.chdir(full_test_setup['model_dir'].replace('/models', ''))

            system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
            system.load_models()
            predictions = system.run_optimized_predictions()

            # Check memory usage after pipeline
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (less than 100MB)
            assert memory_increase < 100 * 1024 * 1024  # 100MB

        finally:
            os.chdir(original_cwd)


class TestErrorRecoveryIntegration:
    """Test error handling and recovery in integrated scenarios."""

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_partial_api_failure_recovery(self, mock_client_class, full_test_setup,
                                        mock_api_key, mock_env_with_api_key):
        """Test recovery from partial API failures."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Create props data with some problematic entries
        props_data = pd.DataFrame({
            'fullName': ['LeBron James', None, 'Kevin Durant'],  # One None value
            'market_type': ['player_points', 'player_points', 'player_points'],
            'line_value': [25.5, 28.5, None],  # One None value
            'over_odds': [-110, -115, -105],
            'bookmaker': ['DraftKings', 'FanDuel', 'DraftKings'],
            'home_team': ['LAL', 'GSW', 'PHX'],
            'away_team': ['GSW', 'PHX', 'LAL'],
            'game_time': [datetime.now(), datetime.now(), datetime.now()],
            'timestamp': [datetime.now(), datetime.now(), datetime.now()]
        })

        mock_client.get_all_todays_player_props.return_value = props_data

        # Mock formatting to handle problematic data
        def format_side_effect(df):
            # Filter out rows with None values
            clean_df = df.dropna()
            return clean_df.rename(columns={
                'player_name': 'fullName',
                'line_value': 'prop_line'
            })

        mock_client.format_for_ml_pipeline.side_effect = format_side_effect

        original_cwd = os.getcwd()
        try:
            os.chdir(full_test_setup['model_dir'].replace('/models', ''))

            system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
            system.load_models()

            # Should handle partial failures gracefully
            predictions = system.run_optimized_predictions()

            # May return None or partial results, but shouldn't crash
            assert predictions is None or isinstance(predictions, list)

        finally:
            os.chdir(original_cwd)

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_corrupted_model_recovery(self, mock_client_class, full_test_setup,
                                     mock_api_key, mock_env_with_api_key):
        """Test recovery from corrupted model files."""
        original_cwd = os.getcwd()
        try:
            os.chdir(full_test_setup['model_dir'].replace('/models', ''))

            # Corrupt one model file
            with open(os.path.join(full_test_setup['model_dir'], 'random_forest_model.pkl'), 'w') as f:
                f.write("corrupted data")

            system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
            result = system.load_models()

            # Should load remaining models successfully
            assert result is True
            assert len(system.models) >= 1  # Should have at least one working model

        finally:
            os.chdir(original_cwd)


class TestConfigurationIntegration:
    """Test configuration and environment integration."""

    @patch.dict(os.environ, {'ODDS_API_KEY': 'env_test_key'})
    def test_environment_variable_integration(self, full_test_setup):
        """Test that environment variables are properly integrated."""
        # Test that environment variable is used
        with patch('final_predictions_optimized.NBAOddsClient'):
            system = OptimizedNBAPredictionsSystem()
            assert system.api_key == 'env_test_key'

    def test_working_directory_integration(self, full_test_setup, mock_api_key):
        """Test that working directory changes don't break the system."""
        original_cwd = os.getcwd()

        try:
            # Change to a different directory
            os.chdir('/tmp')

            with patch('final_predictions_optimized.NBAOddsClient'):
                # Should still work with relative paths
                system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
                assert system.api_key == mock_api_key

        finally:
            os.chdir(original_cwd)