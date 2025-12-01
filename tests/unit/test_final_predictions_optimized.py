"""
Unit Tests for Optimized NBA Predictions System
===============================================

Comprehensive unit tests for src/final_predictions_optimized.py
Tests the production system with proper mocking and edge case coverage.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, mock_open
import joblib

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from final_predictions_optimized import OptimizedNBAPredictionsSystem


class TestOptimizedNBAPredictionsSystemInitialization:
    """Test system initialization and configuration."""

    def test_init_with_api_key(self, mock_api_key):
        """Test initialization with explicit API key."""
        with patch('final_predictions_optimized.NBAOddsClient') as mock_client:
            system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)

            assert system.api_key == mock_api_key
            assert system.models == {}
            assert system.historical_data is None
            mock_client.assert_called_once_with(mock_api_key)

    def test_init_with_env_var(self, mock_env_with_api_key):
        """Test initialization using environment variable."""
        with patch('final_predictions_optimized.NBAOddsClient') as mock_client:
            system = OptimizedNBAPredictionsSystem()

            assert system.api_key == mock_env_with_api_key
            mock_client.assert_called_once_with(mock_env_with_api_key)

    def test_init_without_api_key(self, mock_env_without_api_key):
        """Test initialization fails without API key."""
        with pytest.raises(ValueError, match="API key required"):
            OptimizedNBAPredictionsSystem()

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_init_prints_header(self, mock_client, mock_api_key, capfd):
        """Test that initialization prints system header."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)

        captured = capfd.readouterr()
        assert "OPTIMIZED NBA PREDICTIONS SYSTEM" in captured.out
        assert "Core Features:" in captured.out


class TestModelLoading:
    """Test model loading functionality."""

    @patch('final_predictions_optimized.joblib.load')
    @patch('final_predictions_optimized.os.path.exists')
    @patch('final_predictions_optimized.pd.read_csv')
    @patch('final_predictions_optimized.NBAOddsClient')
    def test_load_models_success(self, mock_client, mock_read_csv, mock_exists, mock_joblib_load,
                                mock_api_key, sample_features_data, temp_model_dir):
        """Test successful model loading."""
        # Setup mocks
        mock_exists.return_value = True
        mock_read_csv.return_value = sample_features_data

        # Mock models
        mock_rf_model = MagicMock()
        mock_xgb_model = MagicMock()

        def joblib_side_effect(filepath):
            if 'random_forest_model.pkl' in filepath:
                return mock_rf_model
            elif 'xgboost_model.pkl' in filepath:
                return mock_xgb_model

        mock_joblib_load.side_effect = joblib_side_effect

        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        result = system.load_models()

        assert result is True
        assert 'random_forest' in system.models
        assert 'xgboost' in system.models
        assert system.historical_data is not None
        assert len(system.models) == 2

    @patch('final_predictions_optimized.os.path.exists')
    @patch('final_predictions_optimized.NBAOddsClient')
    def test_load_models_no_files(self, mock_client, mock_exists, mock_api_key):
        """Test loading when no model files exist."""
        mock_exists.return_value = False

        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        result = system.load_models()

        assert result is False
        assert len(system.models) == 0

    @patch('final_predictions_optimized.joblib.load')
    @patch('final_predictions_optimized.os.path.exists')
    @patch('final_predictions_optimized.pd.read_csv')
    @patch('final_predictions_optimized.NBAOddsClient')
    def test_load_models_missing_data_file(self, mock_client, mock_read_csv, mock_exists,
                                          mock_joblib_load, mock_api_key, temp_model_dir):
        """Test loading when historical data file is missing."""
        mock_exists.side_effect = lambda path: 'model' in path  # Models exist, data doesn't
        mock_joblib_load.return_value = MagicMock()
        mock_read_csv.side_effect = FileNotFoundError("Data file not found")

        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)

        with pytest.raises(FileNotFoundError):
            system.load_models()


class TestPlayerFormAnalysis:
    """Test player form analysis functionality."""

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_analyze_player_form_with_data(self, mock_client, mock_api_key, sample_features_data):
        """Test player form analysis with sufficient data."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        system.historical_data = sample_features_data

        player_name = sample_features_data['fullName'].iloc[0]
        result = system.analyze_player_form(player_name, days_back=15)

        assert 'current_streak' in result
        assert 'streak_length' in result
        assert 'streak_intensity' in result
        assert 'trend' in result
        assert 'form_confidence' in result
        assert 'volatility' in result
        assert result['current_streak'] in ['hot', 'cold', 'neutral']
        assert 0 <= result['form_confidence'] <= 1

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_analyze_player_form_no_data(self, mock_client, mock_api_key, sample_features_data):
        """Test player form analysis with no player data."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        system.historical_data = sample_features_data

        result = system.analyze_player_form("Unknown Player", days_back=15)

        # Should return default form
        assert result['current_streak'] == 'neutral'
        assert result['streak_length'] == 0
        assert result['form_confidence'] == 0.5

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_analyze_player_form_insufficient_games(self, mock_client, mock_api_key, sample_features_data):
        """Test player form analysis with insufficient games for streak detection."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        # Create minimal data
        minimal_data = sample_features_data.head(2).copy()
        system.historical_data = minimal_data

        player_name = minimal_data['fullName'].iloc[0]
        result = system.analyze_player_form(player_name, days_back=15)

        assert result['current_streak'] == 'neutral'
        assert result['trend'] == 0
        assert result['trend_strength'] == 0

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_analyze_player_form_zero_variance(self, mock_client, mock_api_key):
        """Test player form analysis with zero variance (constant points)."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)

        # Create data with constant points
        constant_data = pd.DataFrame({
            'fullName': ['Test Player'] * 5,
            'points': [25] * 5,  # All the same
            'gameDate': pd.date_range('2023-01-01', periods=5)
        })
        system.historical_data = constant_data

        result = system.analyze_player_form('Test Player', days_back=15)

        # Should handle zero variance gracefully
        assert result['form_confidence'] >= 0.3
        assert result['volatility'] == 0


class TestEnhancedMatchupAnalysis:
    """Test enhanced matchup analysis functionality."""

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_analyze_enhanced_matchup_with_data(self, mock_client, mock_api_key, sample_features_data):
        """Test matchup analysis with historical data."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        # Add team names to data
        sample_features_data['playerteamName'] = 'LAL'
        sample_features_data['opponentteamName'] = 'GSW'
        system.historical_data = sample_features_data

        player_name = sample_features_data['fullName'].iloc[0]
        result = system.analyze_enhanced_matchup(player_name, 'GSW', days_back=30)

        assert 'avg_points_vs_opp' in result
        assert 'over_rate_vs_opp' in result
        assert 'efficiency_vs_opp' in result
        assert 'recent_trend_vs_opp' in result
        assert 'consistency_vs_opp' in result
        assert 'sample_confidence' in result
        assert 'matchup_quality' in result
        assert 0 <= result['sample_confidence'] <= 1

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_analyze_enhanced_matchup_no_data(self, mock_client, mock_api_key, sample_features_data):
        """Test matchup analysis with no historical matchup data."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        system.historical_data = sample_features_data

        result = system.analyze_enhanced_matchup("Unknown Player", "Unknown Team", days_back=30)

        # Should return default matchup
        assert result['avg_points_vs_opp'] == 0
        assert result['over_rate_vs_opp'] == 0.5
        assert result['sample_confidence'] == 0
        assert result['matchup_quality'] == 0.5

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_analyze_enhanced_matchup_insufficient_sample(self, mock_client, mock_api_key, sample_features_data):
        """Test matchup analysis with insufficient sample size."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        # Create minimal matchup data
        minimal_data = sample_features_data.head(1).copy()
        minimal_data['playerteamName'] = 'LAL'
        minimal_data['opponentteamName'] = 'GSW'
        system.historical_data = minimal_data

        player_name = minimal_data['fullName'].iloc[0]
        result = system.analyze_enhanced_matchup(player_name, 'GSW', days_back=30)

        # Should have low confidence due to small sample
        assert result['sample_confidence'] < 0.1
        assert result['recent_trend_vs_opp'] == 0

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_analyze_enhanced_matchup_zero_over_rate(self, mock_client, mock_api_key):
        """Test matchup analysis when over rate is zero."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)

        # Create data with zero over rate
        matchup_data = pd.DataFrame({
            'fullName': ['Test Player'] * 3,
            'playerteamName': ['LAL'] * 3,
            'opponentteamName': ['GSW'] * 3,
            'points': [20, 22, 21],
            'numMinutes': [30, 32, 31],
            'over_threshold': [0, 0, 0]  # Never over threshold
        })
        system.historical_data = matchup_data

        result = system.analyze_enhanced_matchup('Test Player', 'GSW', days_back=30)

        assert result['over_rate_vs_opp'] == 0
        assert result['matchup_quality'] == 0.5  # Should default when over_rate is 0


class TestTeamChemistryAnalysis:
    """Test team chemistry analysis functionality."""

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_analyze_team_chemistry_with_data(self, mock_client, mock_api_key, sample_features_data):
        """Test team chemistry analysis with sufficient data."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        # Add team names to data
        sample_features_data['playerteamName'] = 'LAL'
        sample_features_data['opponentteamName'] = 'GSW'
        system.historical_data = sample_features_data

        player_name = sample_features_data['fullName'].iloc[0]
        result = system.analyze_team_chemistry(player_name, 'LAL', days_back=10)

        assert 'team_momentum' in result
        assert 'momentum_consistency' in result
        assert 'team_usage_share' in result
        assert 'chemistry_impact' in result
        assert 0 <= result['team_usage_share'] <= 0.5  # Should be capped at 0.5

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_analyze_team_chemistry_no_team_data(self, mock_client, mock_api_key, sample_features_data):
        """Test team chemistry analysis with no team data."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        system.historical_data = sample_features_data

        result = system.analyze_team_chemistry("Unknown Player", "Unknown Team", days_back=10)

        # Should return default chemistry
        assert result['team_momentum'] == 0
        assert result['momentum_consistency'] == 0.5
        assert result['team_usage_share'] == 0.2

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_analyze_team_chemistry_insufficient_games(self, mock_client, mock_api_key, sample_features_data):
        """Test team chemistry analysis with insufficient games."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        minimal_data = sample_features_data.head(5).copy()
        minimal_data['playerteamName'] = 'LAL'
        minimal_data['opponentteamName'] = 'GSW'
        system.historical_data = minimal_data

        player_name = minimal_data['fullName'].iloc[0]
        result = system.analyze_team_chemistry(player_name, 'LAL', days_back=10)

        assert result['team_momentum'] == 0
        assert result['momentum_consistency'] == 0.5

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_analyze_team_chemistry_zero_team_avg(self, mock_client, mock_api_key):
        """Test team chemistry analysis when team average is zero."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)

        team_data = pd.DataFrame({
            'fullName': ['Test Player', 'Other Player'],
            'playerteamName': ['LAL', 'LAL'],
            'opponentteamName': ['GSW', 'PHX'],
            'points': [20, 0]  # Team average will be 10, but let's make it zero
        })
        system.historical_data = team_data

        # Mock the team average calculation to return zero
        with patch.object(pd.DataFrame, 'mean', return_value=0):
            result = system.analyze_team_chemistry('Test Player', 'LAL', days_back=10)

        assert result['team_usage_share'] == 0.2  # Should default


class TestOptimizedPredictionCalculation:
    """Test the main prediction calculation logic."""

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_calculate_optimized_prediction_success(self, mock_client, mock_api_key, sample_features_data):
        """Test successful prediction calculation."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        system.historical_data = sample_features_data
        sample_features_data['playerteamName'] = 'LAL'
        sample_features_data['opponentteamName'] = 'GSW'

        player_name = sample_features_data['fullName'].iloc[0]
        game_context = {
            'player_team': 'LAL',
            'opponent_team': 'GSW'
        }
        baseline_points = 25.0

        result = system.calculate_optimized_prediction(player_name, game_context, baseline_points)

        assert 'predicted_points' in result
        assert 'confidence_score' in result
        assert 'feature_breakdown' in result
        assert 'adjustments' in result
        assert 'weights' in result

        assert isinstance(result['predicted_points'], (int, float))
        assert 0 <= result['confidence_score'] <= 0.95  # Should be capped at 95%

        # Check weights
        expected_weights = {'form': 0.40, 'matchup': 0.30, 'chemistry': 0.20, 'baseline': 0.10}
        assert result['weights'] == expected_weights

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_calculate_optimized_prediction_adjustment_limits(self, mock_client, mock_api_key,
                                                             sample_features_data):
        """Test that prediction adjustments are properly limited."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        system.historical_data = sample_features_data
        sample_features_data['playerteamName'] = 'LAL'
        sample_features_data['opponentteamName'] = 'GSW'

        player_name = sample_features_data['fullName'].iloc[0]
        game_context = {'player_team': 'LAL', 'opponent_team': 'GSW'}

        # Test with extreme baseline points
        baseline_points = 100.0
        result = system.calculate_optimized_prediction(player_name, game_context, baseline_points)

        # Adjustments should be limited
        assert abs(result['adjustments']['form']) <= 3
        assert abs(result['adjustments']['matchup']) <= 2
        assert abs(result['adjustments']['chemistry']) <= 1.5

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_calculate_optimized_prediction_unknown_player(self, mock_client, mock_api_key, sample_features_data):
        """Test prediction calculation for unknown player."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        system.historical_data = sample_features_data

        game_context = {'player_team': 'LAL', 'opponent_team': 'GSW'}
        result = system.calculate_optimized_prediction("Unknown Player", game_context, 25.0)

        # Should still return a prediction with defaults
        assert 'predicted_points' in result
        assert 'confidence_score' in result
        assert result['confidence_score'] >= 0.1  # Should have minimum confidence


class TestDefaultMethods:
    """Test default value methods."""

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_get_default_form(self, mock_client, mock_api_key):
        """Test default form values."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)

        result = system._get_default_form()

        assert result['current_streak'] == 'neutral'
        assert result['streak_length'] == 0
        assert result['streak_intensity'] == 0
        assert result['trend'] == 0
        assert result['trend_strength'] == 0
        assert result['form_confidence'] == 0.5
        assert result['volatility'] == 1

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_get_default_chemistry(self, mock_client, mock_api_key):
        """Test default chemistry values."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)

        result = system._get_default_chemistry()

        assert result['team_momentum'] == 0
        assert result['momentum_consistency'] == 0.5
        assert result['team_usage_share'] == 0.2
        assert result['chemistry_impact'] == 0.5

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_get_default_matchup(self, mock_client, mock_api_key):
        """Test default matchup values."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)

        result = system._get_default_matchup()

        assert result['avg_points_vs_opp'] == 0
        assert result['over_rate_vs_opp'] == 0.5
        assert result['efficiency_vs_opp'] == 0
        assert result['recent_trend_vs_opp'] == 0
        assert result['consistency_vs_opp'] == 0.5
        assert result['sample_confidence'] == 0
        assert result['matchup_quality'] == 0.5


class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_prediction_with_empty_historical_data(self, mock_client, mock_api_key):
        """Test prediction with empty historical data."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        system.historical_data = pd.DataFrame()

        result = system.analyze_player_form("Any Player")

        # Should return defaults
        assert result['current_streak'] == 'neutral'
        assert result['form_confidence'] == 0.5

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_prediction_with_nan_historical_data(self, mock_client, mock_api_key):
        """Test prediction with NaN values in historical data."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)

        # Create data with NaN values
        nan_data = pd.DataFrame({
            'fullName': ['Test Player'] * 3,
            'points': [20, np.nan, 25],
            'numMinutes': [30, 32, np.nan],
            'gameDate': pd.date_range('2023-01-01', periods=3)
        })
        system.historical_data = nan_data

        result = system.analyze_player_form("Test Player")

        # Should handle NaN values gracefully
        assert 'form_confidence' in result
        assert isinstance(result['form_confidence'], (int, float))

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_confidence_level_mapping(self, mock_client, mock_api_key):
        """Test confidence score to level mapping."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)

        # Test different confidence levels
        assert system._get_confidence_level(0.95) == "Very High"
        assert system._get_confidence_level(0.85) == "High"
        assert system._get_confidence_level(0.75) == "Medium-High"
        assert system._get_confidence_level(0.65) == "Medium"


class TestRunOptimizedPredictions:
    """Test the main prediction pipeline."""

    @patch('final_predictions_optimized.NBAOddsClient')
    @patch.object(OptimizedNBAPredictionsSystem, 'load_models')
    @patch.object(OptimizedNBAPredictionsSystem, 'display_optimized_results')
    def test_run_optimized_predictions_success(self, mock_display, mock_load_models, mock_client,
                                              mock_api_key, sample_prediction_input, sample_features_data,
                                              temp_data_dir, mock_env_with_api_key):
        """Test successful run of optimized predictions."""
        # Setup mocks
        mock_load_models.return_value = True

        with patch('final_predictions_optimized.pd.DataFrame.to_csv') as mock_save:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance

            # Mock API responses
            mock_props_df = sample_prediction_input.copy()
            mock_client_instance.get_all_todays_player_props.return_value = mock_props_df
            mock_client_instance.format_for_ml_pipeline.return_value = mock_props_df

            system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
            system.historical_data = sample_features_data
            sample_features_data['playerteamName'] = 'LAL'

            with patch('final_predictions_optimized.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = '20240115_1200'

                predictions = system.run_optimized_predictions()

                assert predictions is not None
                assert isinstance(predictions, list)
                mock_display.assert_called_once()

    @patch('final_predictions_optimized.NBAOddsClient')
    @patch.object(OptimizedNBAPredictionsSystem, 'load_models')
    def test_run_optimized_predictions_no_models(self, mock_load_models, mock_client, mock_api_key):
        """Test run when model loading fails."""
        mock_load_models.return_value = False

        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        result = system.run_optimized_predictions()

        assert result is None

    @patch('final_predictions_optimized.NBAOddsClient')
    @patch.object(OptimizedNBAPredictionsSystem, 'load_models')
    def test_run_optimized_predictions_no_props(self, mock_load_models, mock_client, mock_api_key,
                                               sample_features_data, mock_env_with_api_key):
        """Test run when no player props are available."""
        mock_load_models.return_value = True

        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_all_todays_player_props.return_value = pd.DataFrame()  # Empty

        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        system.historical_data = sample_features_data

        result = system.run_optimized_predictions()

        assert result is None

    @patch('final_predictions_optimized.NBAOddsClient')
    @patch.object(OptimizedNBAPredictionsSystem, 'load_models')
    def test_run_optimized_predictions_low_confidence(self, mock_load_models, mock_client, mock_api_key,
                                                     sample_features_data, mock_env_with_api_key):
        """Test run when all predictions have low confidence."""
        mock_load_models.return_value = True

        # Create props that will result in low confidence predictions
        low_confidence_props = pd.DataFrame({
            'fullName': ['Unknown Player'],  # Will have no historical data
            'market_type': ['points'],
            'prop_line': [25.5],
            'over_odds': [-110],
            'bookmaker': ['DraftKings'],
            'home_team': ['LAL'],
            'away_team': ['GSW'],
            'game_time': [datetime.now()],
            'gameDate': [datetime.now()]
        })

        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_all_todays_player_props.return_value = low_confidence_props
        mock_client_instance.format_for_ml_pipeline.return_value = low_confidence_props

        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        system.historical_data = sample_features_data

        with patch.object(system, 'display_optimized_results') as mock_display:
            predictions = system.run_optimized_predictions()

            # Should return None if no high-confidence predictions
            assert predictions is None
            mock_display.assert_not_called()


class TestDisplayResults:
    """Test results display functionality."""

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_display_optimized_results_with_data(self, mock_client, mock_api_key, capfd):
        """Test displaying results with prediction data."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)

        predictions = [
            {
                'player_name': 'LeBron James',
                'market_type': 'points',
                'prop_line': 25.5,
                'recommendation': 'OVER',
                'confidence': 0.85,
                'over_odds': -110,
                'bookmaker': 'DraftKings',
                'game_time': datetime.now(),
                'optimized_insights': {
                    'predicted_value': 27.5,
                    'line_diff': 2.0,
                    'confidence_level': 'High',
                    'form': 'hot (85%)',
                    'matchup_quality': '75%',
                    'team_chemistry': '80%',
                    'adjustment': '+2.1'
                }
            }
        ]

        system.display_optimized_results(predictions)

        captured = capfd.readouterr()
        assert 'LeBron James' in captured.out
        assert 'OVER' in captured.out
        assert '85%' in captured.out
        assert 'OPTIMIZED ANALYSIS' in captured.out

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_display_optimized_results_empty(self, mock_client, mock_api_key, capfd):
        """Test displaying empty results."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)

        system.display_optimized_results([])

        captured = capfd.readouterr()
        assert 'No high-confidence predictions' in captured.out

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_display_optimized_results_none(self, mock_client, mock_api_key, capfd):
        """Test displaying None results."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)

        system.display_optimized_results(None)

        captured = capfd.readouterr()
        assert 'No high-confidence predictions' in captured.out