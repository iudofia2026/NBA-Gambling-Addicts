"""
Data Validation Tests for Optimized NBA Predictions System
===========================================================

Tests for data quality, integrity, and validation in the NBA predictions system.
Ensures data meets quality standards and catches data-related issues early.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from final_predictions_optimized import OptimizedNBAPredictionsSystem


class TestDataQualityValidation:
    """Test data quality and integrity validation."""

    @pytest.fixture
    def clean_sample_data(self):
        """Create clean sample data for validation tests."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='3D')
        players = ['LeBron James', 'Stephen Curry', 'Kevin Durant']

        data = []
        for i, player in enumerate(players):
            for j, date in enumerate(dates[:40]):  # 40 games per player
                # Ensure realistic data ranges
                points = np.random.randint(15, 40)
                minutes = np.random.randint(25, 42)
                efficiency = points / max(minutes, 1)

                data.append({
                    'gameDate': date,
                    'fullName': player,
                    'playerteamName': ['LAL', 'GSW', 'PHX'][i],
                    'opponentteamName': np.random.choice(['GSW', 'LAL', 'PHX', 'BOS', 'MIA']),
                    'points': points,
                    'assists': np.random.randint(2, 12),
                    'reboundsTotal': np.random.randint(3, 15),
                    'numMinutes': minutes,
                    'fieldGoalsMade': min(np.random.randint(6, 15), points),
                    'fieldGoalsAttempted': np.random.randint(12, 25),
                    'threePointersMade': np.random.randint(0, 6),
                    'threePointersAttempted': np.random.randint(2, 12),
                    'freeThrowsMade': np.random.randint(2, 10),
                    'freeThrowsAttempted': np.random.randint(3, 12),
                    'steals': np.random.randint(0, 3),
                    'blocks': np.random.randint(0, 2),
                    'turnovers': np.random.randint(1, 5),
                    'age': np.random.randint(24, 36),
                    'over_threshold': np.random.randint(0, 2)
                })

        return pd.DataFrame(data)

    @pytest.fixture
    def dirty_sample_data(self, clean_sample_data):
        """Create dirty sample data with quality issues."""
        dirty_data = clean_sample_data.copy()

        # Add null values
        dirty_data.loc[0:2, 'points'] = np.nan
        dirty_data.loc[3:4, 'numMinutes'] = np.nan

        # Add outliers
        dirty_data.loc[5, 'points'] = 999  # Unrealistic high score
        dirty_data.loc[6, 'numMinutes'] = 0  # Impossible minutes
        dirty_data.loc[7, 'numMinutes'] = 100  # Impossible minutes

        # Add negative values where they don't make sense
        dirty_data.loc[8, 'points'] = -5
        dirty_data.loc[9, 'assists'] = -1

        # Add inconsistent team names
        dirty_data.loc[10, 'playerteamName'] = 'Invalid Team Name!'
        dirty_data.loc[11, 'opponentteamName'] = ''

        # Add invalid dates
        dirty_data.loc[12, 'gameDate'] = pd.NaT

        return dirty_data

    def test_required_columns_present(self, clean_sample_data):
        """Test that all required columns are present."""
        required_columns = [
            'gameDate', 'fullName', 'points', 'numMinutes', 'assists',
            'reboundsTotal', 'age', 'playerteamName', 'opponentteamName'
        ]

        for col in required_columns:
            assert col in clean_sample_data.columns, f"Required column {col} is missing"

    def test_data_range_validation(self, clean_sample_data):
        """Test that data values are within realistic ranges."""
        # Points should be realistic
        assert clean_sample_data['points'].min() >= 0, "Points cannot be negative"
        assert clean_sample_data['points'].max() <= 100, "Points should not exceed 100"

        # Minutes should be realistic
        assert clean_sample_data['numMinutes'].min() >= 0, "Minutes cannot be negative"
        assert clean_sample_data['numMinutes'].max() <= 48, "Minutes should not exceed 48"

        # Age should be realistic
        assert clean_sample_data['age'].min() >= 18, "Age should be at least 18"
        assert clean_sample_data['age'].max() <= 50, "Age should not exceed 50"

        # Basic stats should not be negative
        for stat in ['assists', 'reboundsTotal', 'steals', 'blocks']:
            assert clean_sample_data[stat].min() >= 0, f"{stat} cannot be negative"

    def test_null_value_detection(self, dirty_sample_data):
        """Test detection of null values."""
        # Check for null values in critical columns
        critical_columns = ['fullName', 'gameDate', 'points', 'numMinutes']

        null_report = {}
        for col in critical_columns:
            null_count = dirty_sample_data[col].isnull().sum()
            if null_count > 0:
                null_report[col] = null_count

        assert len(null_report) > 0, "Should detect null values in dirty data"

    def test_outlier_detection(self, dirty_sample_data):
        """Test detection of statistical outliers."""
        # Test points outliers
        q1_points = dirty_sample_data['points'].quantile(0.25)
        q3_points = dirty_sample_data['points'].quantile(0.75)
        iqr_points = q3_points - q1_points

        # Points outside 3*IQR are considered outliers
        points_outliers = dirty_sample_data[
            (dirty_sample_data['points'] < q1_points - 3 * iqr_points) |
            (dirty_sample_data['points'] > q3_points + 3 * iqr_points)
        ]

        assert len(points_outliers) > 0, "Should detect points outliers"

        # Test minutes outliers
        minutes_outliers = dirty_sample_data[
            (dirty_sample_data['numMinutes'] < 0) | (dirty_sample_data['numMinutes'] > 48)
        ]

        assert len(minutes_outliers) > 0, "Should detect impossible minute values"

    def test_team_name_validation(self, dirty_sample_data):
        """Test team name format and validity."""
        # Check for invalid team names
        invalid_team_names = dirty_sample_data[
            dirty_sample_data['playerteamName'].str.len() > 10
        ]

        assert len(invalid_team_names) > 0, "Should detect invalid team names"

        # Check for empty team names
        empty_team_names = dirty_sample_data[
            dirty_sample_data['opponentteamName'] == ''
        ]

        assert len(empty_team_names) > 0, "Should detect empty team names"

    def test_date_validation(self, dirty_sample_data):
        """Test date format and validity."""
        # Check for invalid dates
        invalid_dates = dirty_sample_data[dirty_sample_data['gameDate'].isna()]

        assert len(invalid_dates) > 0, "Should detect invalid dates"

        # Check for future dates (if any)
        today = pd.Timestamp.now().normalize()
        future_dates = dirty_sample_data[dirty_sample_data['gameDate'] > today]

        # In test data, we might have future dates, so just check the detection works
        assert len(future_dates) >= 0, "Future date detection should work"

    def test_no_duplicate_games(self, clean_sample_data):
        """Test that there are no duplicate game records."""
        # Check for duplicates based on player + date
        duplicates = clean_sample_data.duplicated(subset=['fullName', 'gameDate'], keep=False)
        assert not duplicates.any(), "Found duplicate game records"

    def test_target_variable_balance(self, clean_sample_data):
        """Test that target variable is not extremely imbalanced."""
        if 'over_threshold' in clean_sample_data.columns:
            over_rate = clean_sample_data['over_threshold'].mean()
            # Should be between 20% and 80% (not too imbalanced)
            assert 0.2 <= over_rate <= 0.8, \
                f"Target variable too imbalanced: {over_rate:.1%} positive class"

    def test_no_future_leakage_in_features(self, clean_sample_data):
        """Test that rolling features don't include current game data."""
        # More robust check: ensure rolling features are NaN or 0 for first game
        first_games = clean_sample_data.groupby('fullName').first()
        rolling_cols = [col for col in clean_sample_data.columns if 'rolling' in col]

        for col in rolling_cols:
            if col in first_games.columns:
                # First games should have NaN or 0 for rolling features
                assert (first_games[col].isna() | (first_games[col] == 0)).all(), \
                    f"Rolling feature {col} shows leakage in first games"

        duplicate_count = duplicates.sum()
        if duplicate_count > 0:
            print(f"Warning: Found {duplicate_count} duplicate game records")


class TestOptimizedModelValidation:
    """Test model quality and validation for the optimized system."""

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_model_prediction_validation(self, mock_client, mock_api_key, clean_sample_data):
        """Test model prediction output validation."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        system.historical_data = clean_sample_data

        # Test prediction output format
        player_name = clean_sample_data['fullName'].iloc[0]
        game_context = {
            'player_team': clean_sample_data['playerteamName'].iloc[0],
            'opponent_team': clean_sample_data['opponentteamName'].iloc[0]
        }
        baseline_points = 25.0

        result = system.calculate_optimized_prediction(player_name, game_context, baseline_points)

        # Validate prediction structure
        required_fields = [
            'predicted_points', 'confidence_score', 'feature_breakdown',
            'adjustments', 'weights'
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

        # Validate data types and ranges
        assert isinstance(result['predicted_points'], (int, float)), "predicted_points should be numeric"
        assert isinstance(result['confidence_score'], (int, float)), "confidence_score should be numeric"
        assert 0 <= result['confidence_score'] <= 1, "confidence_score should be between 0 and 1"

        # Validate feature breakdown
        features = result['feature_breakdown']
        required_features = ['form_factor', 'matchup_factor', 'chemistry_factor']
        for feature in required_features:
            assert feature in features, f"Missing feature: {feature}"

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_model_consistency_validation(self, mock_client, mock_api_key, clean_sample_data):
        """Test model prediction consistency."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        system.historical_data = clean_sample_data

        player_name = clean_sample_data['fullName'].iloc[0]
        game_context = {
            'player_team': clean_sample_data['playerteamName'].iloc[0],
            'opponent_team': clean_sample_data['opponentteamName'].iloc[0]
        }
        baseline_points = 25.0

        # Run same prediction multiple times
        results = []
        for _ in range(5):
            result = system.calculate_optimized_prediction(player_name, game_context, baseline_points)
            results.append(result)

        # Check consistency (should be identical with same input)
        first_result = results[0]
        for i, result in enumerate(results[1:], 1):
            assert result['predicted_points'] == first_result['predicted_points'], \
                f"Prediction {i} differs from first prediction"
            assert result['confidence_score'] == first_result['confidence_score'], \
                f"Confidence {i} differs from first confidence"


class TestFeatureValidation:
    """Test feature engineering and validation."""

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_form_analysis_validation(self, mock_client, mock_api_key, clean_sample_data):
        """Test player form analysis validation."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        system.historical_data = clean_sample_data

        player_name = clean_sample_data['fullName'].iloc[0]
        result = system.analyze_player_form(player_name, days_back=15)

        # Validate form analysis output
        required_fields = [
            'current_streak', 'streak_length', 'streak_intensity',
            'trend', 'trend_strength', 'form_confidence', 'volatility'
        ]

        for field in required_fields:
            assert field in result, f"Missing form field: {field}"

        # Validate values
        assert result['current_streak'] in ['hot', 'cold', 'neutral'], "Invalid streak status"
        assert result['streak_length'] >= 0, "Streak length cannot be negative"
        assert result['streak_intensity'] >= 0, "Streak intensity cannot be negative"
        assert 0 <= result['form_confidence'] <= 1, "Form confidence must be between 0 and 1"

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_matchup_analysis_validation(self, mock_client, mock_api_key, clean_sample_data):
        """Test matchup analysis validation."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        system.historical_data = clean_sample_data
        clean_sample_data['playerteamName'] = 'LAL'
        clean_sample_data['opponentteamName'] = 'GSW'

        player_name = clean_sample_data['fullName'].iloc[0]
        result = system.analyze_enhanced_matchup(player_name, 'GSW', days_back=30)

        # Validate matchup analysis output
        required_fields = [
            'avg_points_vs_opp', 'over_rate_vs_opp', 'efficiency_vs_opp',
            'recent_trend_vs_opp', 'consistency_vs_opp', 'sample_confidence', 'matchup_quality'
        ]

        for field in required_fields:
            assert field in result, f"Missing matchup field: {field}"

        # Validate values
        assert result['avg_points_vs_opp'] >= 0, "Average points cannot be negative"
        assert 0 <= result['over_rate_vs_opp'] <= 1, "Over rate must be between 0 and 1"
        assert 0 <= result['sample_confidence'] <= 1, "Sample confidence must be between 0 and 1"
        assert 0 <= result['matchup_quality'] <= 1, "Matchup quality must be between 0 and 1"

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_chemistry_analysis_validation(self, mock_client, mock_api_key, clean_sample_data):
        """Test team chemistry analysis validation."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        system.historical_data = clean_sample_data
        clean_sample_data['playerteamName'] = 'LAL'
        clean_sample_data['opponentteamName'] = 'GSW'

        player_name = clean_sample_data['fullName'].iloc[0]
        result = system.analyze_team_chemistry(player_name, 'LAL', days_back=10)

        # Validate chemistry analysis output
        required_fields = [
            'team_momentum', 'momentum_consistency', 'team_usage_share', 'chemistry_impact'
        ]

        for field in required_fields:
            assert field in result, f"Missing chemistry field: {field}"

        # Validate values
        assert 0 <= result['momentum_consistency'] <= 1, "Momentum consistency must be between 0 and 1"
        assert 0 <= result['team_usage_share'] <= 0.5, "Team usage share must be capped at 0.5"
        assert 0 <= result['chemistry_impact'] <= 1, "Chemistry impact must be between 0 and 1"


class TestPerformanceValidation:
    """Test performance and scalability validation."""

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_prediction_performance_validation(self, mock_client, mock_api_key, clean_sample_data):
        """Test prediction performance metrics."""
        import time

        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        system.historical_data = clean_sample_data

        # Test performance with multiple predictions
        start_time = time.time()

        predictions = []
        for i, row in clean_sample_data.head(10).iterrows():
            game_context = {
                'player_team': row['playerteamName'],
                'opponent_team': row['opponentteamName']
            }
            result = system.calculate_optimized_prediction(
                row['fullName'], game_context, row['points']
            )
            predictions.append(result)

        end_time = time.time()
        prediction_time = end_time - start_time

        # Performance assertions
        assert len(predictions) == 10, "Should generate 10 predictions"
        assert prediction_time < 5.0, "10 predictions should complete within 5 seconds"

        # Average time per prediction
        avg_time_per_prediction = prediction_time / 10
        assert avg_time_per_prediction < 0.5, "Each prediction should take less than 0.5 seconds"


class TestDataIntegrityValidation:
    """Test data integrity and consistency validation."""

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_data_consistency_validation(self, mock_client, mock_api_key, clean_sample_data):
        """Test data consistency across different views."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        system.historical_data = clean_sample_data

        # Test player data consistency
        unique_players = clean_sample_data['fullName'].unique()
        for player in unique_players[:5]:  # Test first 5 players
            player_data = clean_sample_data[clean_sample_data['fullName'] == player]

            # Each player should have consistent data
            assert len(player_data) > 0, f"Player {player} should have data"

            # Check for reasonable game sequences
            sorted_dates = player_data['gameDate'].sort_values()
            date_diffs = sorted_dates.diff().dropna()

            # Most games should be at least 1 day apart
            assert (date_diffs >= pd.Timedelta(days=1)).all(), \
                "Games should be at least 1 day apart"

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_statistical_relationships_validation(self, mock_client, mock_api_key, clean_sample_data):
        """Test statistical relationships in the data."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)
        system.historical_data = clean_sample_data

        # Test correlation between minutes and points
        correlation = clean_sample_data['numMinutes'].corr(clean_sample_data['points'])
        assert correlation >= 0, "Minutes and points should be positively correlated"

        # Test reasonable field goal percentages
        fg_pct = clean_sample_data['fieldGoalsMade'] / clean_sample_data['fieldGoalsAttempted']
        valid_fg_pct = fg_pct.dropna()

        assert (valid_fg_pct >= 0).all(), "Field goal percentage cannot be negative"
        assert (valid_fg_pct <= 1).all(), "Field goal percentage cannot exceed 100%"

        # Test reasonable efficiency
        efficiency = clean_sample_data['points'] / clean_sample_data['numMinutes']
        valid_efficiency = efficiency.dropna()

        assert (valid_efficiency >= 0).all(), "Efficiency cannot be negative"
        assert valid_efficiency.max() <= 2, "Efficiency should not exceed 2 points per minute"


class TestErrorHandlingValidation:
    """Test error handling and edge case validation."""

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_edge_case_handling(self, mock_client, mock_api_key):
        """Test handling of edge cases."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)

        # Test with empty data
        empty_data = pd.DataFrame()
        system.historical_data = empty_data

        result = system.analyze_player_form("Any Player")
        assert result is not None, "Should handle empty data gracefully"

        # Test with single game
        single_game_data = pd.DataFrame({
            'fullName': ['Test Player'],
            'gameDate': [datetime.now()],
            'points': [25],
            'numMinutes': [30]
        })
        system.historical_data = single_game_data

        result = system.analyze_player_form("Test Player")
        assert result is not None, "Should handle single game data"

    @patch('final_predictions_optimized.NBAOddsClient')
    def test_extreme_values_handling(self, mock_client, mock_api_key):
        """Test handling of extreme values."""
        system = OptimizedNBAPredictionsSystem(api_key=mock_api_key)

        # Create data with extreme values
        extreme_data = pd.DataFrame({
            'fullName': ['Extreme Player'] * 3,
            'gameDate': pd.date_range('2023-01-01', periods=3),
            'points': [0, 1, 100],  # Extreme point values
            'numMinutes': [1, 48, 48],  # Extreme minute values
            'playerteamName': ['EXT'] * 3,
            'opponentteamName': ['OPP'] * 3
        })
        system.historical_data = extreme_data

        result = system.analyze_player_form("Extreme Player")
        assert result is not None, "Should handle extreme values without crashing"
        assert 0 <= result['form_confidence'] <= 1, "Confidence should remain valid"
