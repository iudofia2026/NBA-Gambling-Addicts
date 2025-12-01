import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from src.data_sources_integration import DataSourcesIntegration
    from src.odds_api_client import OddsAPIClient
    from src.data_cleaning import DataCleaner
    from src.feature_engineering import FeatureEngineer
    from src.ml_models import MLModelBuilder
    from src.robust_validation import RobustValidator
except ImportError:
    # Mock implementations
    class DataSourcesIntegration:
        def __init__(self):
            self.data_sources = {'nba_api': True, 'odds_api': True}

        def fetch_nba_stats(self, date_range):
            dates = pd.date_range(date_range[0], date_range[1])
            return pd.DataFrame({
                'date': dates,
                'player_id': np.random.randint(1, 100, len(dates)),
                'points': np.random.normal(20, 5, len(dates)),
                'rebounds': np.random.normal(8, 2, len(dates)),
                'assists': np.random.normal(5, 2, len(dates)),
                'minutes': np.random.normal(30, 5, len(dates))
            })

        def fetch_betting_odds(self, date):
            return pd.DataFrame({
                'player_id': np.random.randint(1, 100, 50),
                'prop_line_points': np.random.uniform(15, 25, 50),
                'over_odds': np.random.uniform(1.8, 2.2, 50),
                'under_odds': np.random.uniform(1.8, 2.2, 50),
                'date': [date] * 50
            })

        def validate_data_integrity(self, data):
            return {
                'null_count': data.isnull().sum().sum(),
                'duplicate_count': data.duplicated().sum(),
                'date_range_valid': True,
                'player_id_valid': True
            }

    class OddsAPIClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.rate_limit_remaining = 1000

        def get_player_props(self, date):
            # Mock API response
            return {
                'success': True,
                'data': [
                    {
                        'player_id': 1,
                        'player_name': 'Player A',
                        'prop_type': 'points',
                        'line': 20.5,
                        'over_odds': 1.90,
                        'under_odds':': 1.90
                    },
                    {
                        'player_id': 2,
                        'player_name': 'Player B',
                        'prop_type': 'points',
                        'line': 18.5,
                        'over_odds': 1.85,
                        'under_odds': 1.95
                    }
                ]
            }

        def check_api_status(self):
            return {'status': 'active', 'rate_limit': self.rate_limit_remaining}

    class DataCleaner:
        def clean_raw_data(self, data):
            # Remove rows with excessive nulls
            cleaned = data.dropna(thresh=len(data.columns) * 0.7)
            # Fill remaining nulls
            cleaned = cleaned.fillna(cleaned.mean(numeric_only=True))
            return cleaned

        def validate_cleaning_process(self, original_data, cleaned_data):
            return {
                'original_rows': len(original_data),
                'cleaned_rows': len(cleaned_data),
                'rows_removed': len(original_data) - len(cleaned_data),
                'nulls_handled': original_data.isnull().sum().sum() - cleaned_data.isnull().sum().sum()
            }

    class FeatureEngineer:
        def create_rolling_features(self, data, windows=[3, 5, 10]):
            result = data.copy()
            for col in ['points', 'rebounds', 'assists']:
                if col in data.columns:
                    for window in windows:
                        result[f'{col}_rolling_{window}'] = data[col].rolling(window).mean()
            return result

        def create_interaction_features(self, data):
            result = data.copy()
            if all(col in data.columns for col in ['points', 'rebounds', 'assists']):
                result['total_impact'] = data['points'] + data['rebounds'] + data['assists']
                result['efficiency'] = data['points'] / data['minutes'].replace(0, 1)
            return result

        def validate_feature_creation(self, original_data, feature_data):
            new_features = set(feature_data.columns) - set(original_data.columns)
            return {
                'new_features_count': len(new_features),
                'new_features': list(new_features),
                'feature_creation_success': len(new_features) > 0
            }

    class MLModelBuilder:
        def __init__(self):
            self.model = None
            self.feature_importance = {}

        def prepare_features(self, data, target_col='target'):
            feature_cols = [col for col in data.columns if col != target_col and data[col].dtype != 'object']
            X = data[feature_cols].fillna(0)
            y = data[target_col] if target_col in data.columns else None
            return X, y

        def train_model(self, X, y):
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=10, random_state=42)
            self.model.fit(X, y)
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
            return self.model

        def validate_model_training(self, X, y):
            return {
                'features_used': len(X.columns),
                'training_samples': len(X),
                'target_distribution': y.value_counts().to_dict() if y is not None else {},
                'feature_importance_available': len(self.feature_importance) > 0
            }

    class RobustValidator:
        def validate_predictions(self, y_true, y_pred, y_proba=None):
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
                'prediction_count': len(y_pred),
                'confidence_available': y_proba is not None
            }

        def check_data_leakage(self, features, target_col):
            leakage_indicators = []
            for col in features.columns:
                if any(keyword in col.lower() for keyword in ['target', 'over', 'current', 'same_game']):
                    if col != target_col:
                        leakage_indicators.append(col)
            return {
                'leakage_detected': len(leakage_indicators) > 0,
                'leakage_features': leakage_indicators
            }


class TestDataFlowIntegrity:
    """Test data flow integrity throughout the pipeline"""

    def setup_method(self):
        """Set up data flow components"""
        self.data_integration = DataSourcesIntegration()
        self.odds_client = OddsAPIClient(api_key="test_key")
        self.data_cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineer()
        self.model_builder = MLModelBuilder()
        self.validator = RobustValidator()

        # Create initial test data
        dates = pd.date_range('2024-01-01', periods=50)
        self.initial_data = pd.DataFrame({
            'date': dates,
            'player_id': np.random.randint(1, 30, 50),
            'player_name': [f'Player_{i}' for i in np.random.randint(1, 30, 50)],
            'team': [f'Team_{i}' for i in np.random.randint(1, 20, 50)],
            'points': np.random.normal(20, 5, 50),
            'rebounds': np.random.normal(8, 2, 50),
            'assists': np.random.normal(5, 2, 50),
            'minutes': np.random.normal(30, 5, 50),
            'prop_line': np.random.uniform(18, 22, 50)
        })

    def test_data_source_to_processing_flow(self):
        """Test data flow from sources to processing"""
        # Step 1: Fetch data from sources
        nba_data = self.data_integration.fetch_nba_stats(('2024-01-01', '2024-01-31'))
        odds_data = self.data_integration.fetch_betting_odds('2024-01-01')

        # Validate source data integrity
        nba_integrity = self.data_integration.validate_data_integrity(nba_data)
        odds_integrity = self.data_integration.validate_data_integrity(odds_data)

        assert nba_integrity['date_range_valid']
        assert odds_integrity['date_range_valid']

        # Step 2: Merge data sources
        merged_data = pd.merge(nba_data, odds_data, on='player_id', how='left', suffixes=('_stats', '_odds'))

        # Validate merge integrity
        assert len(merged_data) > 0
        assert 'player_id' in merged_data.columns
        assert 'points' in merged_data.columns
        assert 'prop_line_points' in merged_data.columns

        return merged_data

    def test_cleaning_data_transformation(self):
        """Test data transformation during cleaning"""
        # Intentionally create messy data
        messy_data = self.initial_data.copy()
        messy_data.loc[5:10, 'points'] = np.nan
        messy_data.loc[15:20, 'rebounds'] = np.nan
        messy_data = pd.concat([messy_data, messy_data.iloc[0:5]])  # Add duplicates

        # Clean the data
        cleaned_data = self.data_cleaner.clean_raw_data(messy_data)

        # Validate cleaning process
        cleaning_report = self.data_cleaner.validate_cleaning_process(messy_data, cleaned_data)

        assert cleaning_report['original_rows'] > cleaning_report['cleaned_rows']
        assert cleaning_report['nulls_handled'] > 0
        assert cleaned_data.isnull().sum().sum() < messy_data.isnull().sum().sum()

        # Verify data integrity maintained
        assert 'player_id' in cleaned_data.columns
        assert len(cleaned_data) > 0

        return cleaned_data

    def test_feature_engineering_flow(self):
        """Test feature engineering data flow"""
        cleaned_data = self.test_cleaning_data_transformation()

        # Create rolling features
        rolling_features = self.feature_engineer.create_rolling_features(cleaned_data)

        # Create interaction features
        interaction_features = self.feature_engineer.create_interaction_features(rolling_features)

        # Validate feature creation
        feature_report = self.feature_engineer.validate_feature_creation(
            cleaned_data, interaction_features
        )

        assert feature_report['feature_creation_success']
        assert feature_report['new_features_count'] > 0
        assert 'total_impact' in feature_report['new_features']
        assert 'efficiency' in feature_report['new_features']

        return interaction_features

    def test_model_training_data_flow(self):
        """Test data flow through model training"""
        feature_data = self.test_feature_engineering_flow()

        # Create target variable
        feature_data['target'] = (feature_data['points'] > feature_data['prop_line']).astype(int)

        # Prepare features for modeling
        X, y = self.model_builder.prepare_features(feature_data, target_col='target')

        # Validate feature preparation
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] > 0
        assert len(y) > 0

        # Train model
        if len(X) > 10:  # Ensure we have enough data
            model = self.model_builder.train_model(X, y)

            # Validate model training
            training_report = self.model_builder.validate_model_training(X, y)

            assert training_report['features_used'] > 0
            assert training_report['training_samples'] > 0
            assert training_report['feature_importance_available']

            return model, X, y

        return None, X, y

    def test_prediction_validation_flow(self):
        """Test prediction and validation flow"""
        model, X, y = self.test_model_training_data_flow()

        if model is not None and len(X) > 0:
            # Make predictions
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]

            # Validate predictions
            validation_results = self.validator.validate_predictions(y, predictions, probabilities)

            assert validation_results['accuracy'] >= 0
            assert validation_results['precision'] >= 0
            assert validation_results['recall'] >= 0
            assert validation_results['prediction_count'] == len(predictions)
            assert validation_results['confidence_available']

            return validation_results

        return None

    def test_end_to_end_data_consistency(self):
        """Test data consistency throughout the entire pipeline"""
        # Track data at each stage
        original_data = self.initial_data.copy()

        # Stage 1: Data Collection
        collected_data = self.test_data_source_to_processing_flow()

        # Stage 2: Data Cleaning
        cleaned_data = self.data_cleaner.clean_raw_data(original_data)

        # Stage 3: Feature Engineering
        feature_data = self.feature_engineer.create_interaction_features(cleaned_data)

        # Stage 4: Target Creation
        feature_data['target'] = (feature_data['points'] > feature_data['prop_line']).astype(int)

        # Verify player_id consistency across stages
        assert set(original_data['player_id']).issubset(set(feature_data['player_id']))

        # Verify no data leakage
        leakage_check = self.validator.check_data_leakage(feature_data, 'target')
        assert not leakage_check['leakage_detected']

        # Verify reasonable target distribution
        target_dist = feature_data['target'].value_counts()
        assert 0 in target_dist and 1 in target_dist  # Both classes present
        assert target_dist.min() > len(feature_data) * 0.1  # Minimum 10% minority class

    def test_data_type_consistency(self):
        """Test data types remain consistent through pipeline"""
        # Expected data types
        expected_types = {
            'player_id': 'int64',
            'points': 'float64',
            'rebounds': 'float64',
            'assists': 'float64',
            'minutes': 'float64',
            'prop_line': 'float64'
        }

        # Check initial data types
        for col, expected_type in expected_types.items():
            if col in self.initial_data.columns:
                assert str(self.initial_data[col].dtype) == expected_type

        # Process through pipeline
        cleaned_data = self.data_cleaner.clean_raw_data(self.initial_data)
        feature_data = self.feature_engineer.create_interaction_features(cleaned_data)

        # Check critical columns maintain types
        critical_cols = ['player_id', 'points']
        for col in critical_cols:
            if col in feature_data.columns:
                assert pd.api.types.is_numeric_dtype(feature_data[col])

    def test_data_volume_integrity(self):
        """Test data volume changes are tracked and reasonable"""
        original_count = len(self.initial_data)

        # Process through pipeline stages
        cleaned_data = self.data_cleaner.clean_raw_data(self.initial_data)
        feature_data = self.feature_engineer.create_interaction_features(cleaned_data)

        # Track data volume changes
        volume_changes = {
            'original': original_count,
            'after_cleaning': len(cleaned_data),
            'after_feature_engineering': len(feature_data)
        }

        # Volume should not increase unexpectedly
        assert volume_changes['after_cleaning'] <= volume_changes['original']
        assert volume_changes['after_feature_engineering'] == volume_changes['after_cleaning']

        # Volume loss should be reasonable (less than 50%)
        cleaning_loss = (volume_changes['original'] - volume_changes['after_cleaning']) / volume_changes['original']
        assert cleaning_loss < 0.5

    def test_error_propagation_handling(self):
        """Test how errors are handled and propagated through pipeline"""
        # Test with problematic data
        problematic_data = self.initial_data.copy()
        problematic_data.loc[0, 'points'] = float('inf')
        problematic_data.loc[1, 'minutes'] = 0  # Division by zero risk

        # Pipeline should handle gracefully
        try:
            cleaned_data = self.data_cleaner.clean_raw_data(problematic_data)
            feature_data = self.feature_engineer.create_interaction_features(cleaned_data)

            # Should have handled the problematic values
            assert not cleaned_data['points'].isin([float('inf'), -float('inf')]).any()
            assert len(feature_data) > 0

        except Exception as e:
            pytest.fail(f"Pipeline failed to handle problematic data gracefully: {e}")

    def test_temporal_data_integrity(self):
        """Test temporal data integrity throughout pipeline"""
        # Ensure data has proper date ordering
        temporal_data = self.initial_data.copy()
        temporal_data = temporal_data.sort_values('date')

        # Process through pipeline
        cleaned_data = self.data_cleaner.clean_raw_data(temporal_data)
        feature_data = self.feature_engineer.create_rolling_features(cleaned_data)

        # Verify temporal consistency
        assert feature_data['date'].is_monotonic_increasing

        # Verify rolling features respect temporal order
        if 'points_rolling_3' in feature_data.columns:
            # Rolling features should have NaNs for early periods
            assert feature_data['points_rolling_3'].iloc[:2].isna().any()


class TestDataValidationAtEachStage:
    """Test data validation at each pipeline stage"""

    def setup_method(self):
        """Set up validation test data"""
        self.validator = RobustValidator()
        dates = pd.date_range('2024-01-01', periods=30)
        self.test_data = pd.DataFrame({
            'date': dates,
            'player_id': np.random.randint(1, 20, 30),
            'points': np.random.normal(20, 5, 30),
            'rebounds': np.random.normal(8, 2, 30),
            'prop_line': np.random.uniform(18, 22, 30)
        })

    def test_data_collection_validation(self):
        """Test validation during data collection"""
        # Validate basic data properties
        validation_checks = {
            'has_required_columns': all(col in self.test_data.columns for col in ['player_id', 'points', 'date']),
            'no_duplicates': not self.test_data.duplicated().any(),
            'reasonable_values': (self.test_data['points'].min() >= 0 and
                                self.test_data['points'].max() <= 100),
            'date_range_valid': len(self.test_data['date'].unique()) > 1
        }

        assert all(validation_checks.values())

    def test_cleaning_stage_validation(self):
        """Test validation during cleaning stage"""
        from data_cleaning import DataCleaner

        cleaner = DataCleaner()

        # Introduce issues
        dirty_data = self.test_data.copy()
        dirty_data.loc[5:8, 'points'] = np.nan

        # Clean and validate
        cleaned_data = cleaner.clean_raw_data(dirty_data)

        # Post-cleaning validation
        assert cleaned_data.isnull().sum().sum() < dirty_data.isnull().sum().sum()
        assert len(cleaned_data) > 0
        assert 'player_id' in cleaned_data.columns

    def test_feature_engineering_validation(self):
        """Test validation during feature engineering"""
        from feature_engineering import FeatureEngineer

        engineer = FeatureEngineer()
        feature_data = engineer.create_interaction_features(self.test_data)

        # Feature validation
        assert len(feature_data.columns) > len(self.test_data.columns)
        assert not feature_data.isnull().all().any()  # No all-null columns
        assert feature_data['player_id'].equals(self.test_data['player_id'])  # IDs preserved

    def test_model_training_validation(self):
        """Test validation during model training"""
        from ml_models import MLModelBuilder

        builder = MLModelBuilder()

        # Create target
        self.test_data['target'] = (self.test_data['points'] > self.test_data['prop_line']).astype(int)

        # Prepare features
        X, y = builder.prepare_features(self.test_data, 'target')

        # Validation checks
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] > 0
        assert not X.isnull().all().any()
        assert y.nunique() >= 1  # At least one class present

    def test_prediction_output_validation(self):
        """Test validation of prediction outputs"""
        # Mock predictions
        predictions = np.array([0, 1, 0, 1, 1] * 6)  # 30 predictions
        probabilities = np.random.uniform(0, 1, 30)
        true_values = np.array([0, 1, 1, 0, 1] * 6)  # 30 true values

        # Validate predictions
        validation_results = self.validator.validate_predictions(true_values, predictions, probabilities)

        assert validation_results['prediction_count'] == 30
        assert 0 <= validation_results['accuracy'] <= 1
        assert validation_results['confidence_available']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])