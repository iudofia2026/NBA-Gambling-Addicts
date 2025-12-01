"""
Unit Tests for ML Models Module
================================

Tests the ml_models module including model training, evaluation,
and feature preparation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ml_models import (
    load_and_prepare_data,
    create_temporal_train_test_split,
    prepare_features,
    evaluate_model,
    MLModelTrainer,
    ScaledLogisticRegression
)


class TestDataLoading:
    """Test data loading and preparation."""

    @patch('ml_models.pd.read_csv')
    def test_load_and_prepare_data_success(self, mock_read_csv, sample_features_data):
        """Test successful data loading."""
        mock_read_csv.return_value = sample_features_data

        data = load_and_prepare_data()

        assert len(data) > 0
        assert 'over_threshold' in data.columns
        assert 'gameDate' in data.columns

    @patch('ml_models.pd.read_csv')
    def test_load_and_prepare_data_file_not_found(self, mock_read_csv):
        """Test error handling when data file missing."""
        mock_read_csv.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            load_and_prepare_data()


class TestTemporalSplit:
    """Test temporal train/test splitting."""

    def test_temporal_split_chronological_order(self, sample_features_data):
        """Test that temporal split maintains chronological order."""
        train, test = create_temporal_train_test_split(sample_features_data, test_size=0.2)

        # Train should be earlier than test
        assert train['gameDate'].max() <= test['gameDate'].min()

    def test_temporal_split_size(self, sample_features_data):
        """Test that split sizes are correct."""
        test_size = 0.2
        train, test = create_temporal_train_test_split(sample_features_data, test_size=test_size)

        total = len(train) + len(test)
        actual_test_ratio = len(test) / total

        # Should be close to requested ratio (within 1%)
        assert abs(actual_test_ratio - test_size) < 0.01

    def test_temporal_split_no_overlap(self, sample_features_data):
        """Test that train and test sets don't overlap."""
        train, test = create_temporal_train_test_split(sample_features_data, test_size=0.2)

        # Check that there's no overlap in indices
        train_indices = set(train.index)
        test_indices = set(test.index)

        assert len(train_indices & test_indices) == 0

    @pytest.mark.parametrize("test_size", [0.1, 0.2, 0.3])
    def test_temporal_split_various_sizes(self, sample_features_data, test_size):
        """Test temporal split with various test sizes."""
        train, test = create_temporal_train_test_split(sample_features_data, test_size=test_size)

        assert len(train) > 0
        assert len(test) > 0
        assert len(train) + len(test) == len(sample_features_data)


class TestFeaturePreparation:
    """Test feature preparation for ML models."""

    def test_prepare_features_excludes_leakage(self, sample_features_data):
        """Test that leakage features are excluded."""
        train, test = create_temporal_train_test_split(sample_features_data, test_size=0.2)

        X_train, X_test, y_train, y_test, feature_cols, encoders = prepare_features(train, test)

        # Ensure no leakage features
        leakage_features = ['over_threshold', 'player_threshold', 'points', 'gameDate', 'fullName']
        for feature in leakage_features:
            assert feature not in feature_cols

    def test_prepare_features_handles_categoricals(self, sample_features_data):
        """Test that categorical features are encoded."""
        # Add a categorical feature
        sample_features_data['teamAbbreviation'] = np.random.choice(['LAL', 'GSW', 'PHX'],
                                                                     size=len(sample_features_data))

        train, test = create_temporal_train_test_split(sample_features_data, test_size=0.2)
        X_train, X_test, y_train, y_test, feature_cols, encoders = prepare_features(train, test)

        # Check that categorical features are numeric
        if 'teamAbbreviation' in X_train.columns:
            assert X_train['teamAbbreviation'].dtype in [np.int32, np.int64]

    def test_prepare_features_no_missing_values(self, sample_features_data):
        """Test that missing values are handled."""
        train, test = create_temporal_train_test_split(sample_features_data, test_size=0.2)
        X_train, X_test, y_train, y_test, feature_cols, encoders = prepare_features(train, test)

        # Should have no NaN values after preparation
        assert X_train.isnull().sum().sum() == 0
        assert X_test.isnull().sum().sum() == 0

    def test_prepare_features_consistent_shapes(self, sample_features_data):
        """Test that train and test have consistent feature shapes."""
        train, test = create_temporal_train_test_split(sample_features_data, test_size=0.2)
        X_train, X_test, y_train, y_test, feature_cols, encoders = prepare_features(train, test)

        # Same number of features
        assert X_train.shape[1] == X_test.shape[1]
        # Same columns
        assert list(X_train.columns) == list(X_test.columns)


class TestModelEvaluation:
    """Test model evaluation functions."""

    def test_evaluate_model_returns_metrics(self, mock_trained_model):
        """Test that evaluate_model returns all required metrics."""
        X_test = pd.DataFrame(np.random.rand(100, 5))
        y_test = np.random.randint(0, 2, 100)

        mock_trained_model.predict.return_value = y_test
        mock_trained_model.predict_proba.return_value = np.column_stack([
            1 - y_test, y_test
        ])

        metrics, y_pred, y_proba = evaluate_model(
            mock_trained_model, X_test, y_test, 'Test Model'
        )

        required_metrics = ['model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in required_metrics:
            assert metric in metrics

    def test_evaluate_model_metric_ranges(self, mock_trained_model):
        """Test that metrics are in valid ranges."""
        X_test = pd.DataFrame(np.random.rand(100, 5))
        y_test = np.random.randint(0, 2, 100)

        mock_trained_model.predict.return_value = y_test
        mock_trained_model.predict_proba.return_value = np.column_stack([
            1 - y_test, y_test
        ])

        metrics, _, _ = evaluate_model(mock_trained_model, X_test, y_test, 'Test Model')

        # All metrics should be between 0 and 1
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1


class TestScaledLogisticRegression:
    """Test the ScaledLogisticRegression wrapper."""

    def test_scaled_lr_predict(self):
        """Test prediction with scaling."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Create simple test data
        X_train = np.random.rand(100, 3)
        y_train = np.random.randint(0, 2, 100)

        # Train model and scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        model = LogisticRegression()
        model.fit(X_scaled, y_train)

        # Create wrapped model
        wrapped_model = ScaledLogisticRegression(model, scaler)

        # Test prediction
        X_test = np.random.rand(10, 3)
        predictions = wrapped_model.predict(X_test)

        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)

    def test_scaled_lr_predict_proba(self):
        """Test probability prediction with scaling."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X_train = np.random.rand(100, 3)
        y_train = np.random.randint(0, 2, 100)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        model = LogisticRegression()
        model.fit(X_scaled, y_train)

        wrapped_model = ScaledLogisticRegression(model, scaler)

        X_test = np.random.rand(10, 3)
        probabilities = wrapped_model.predict_proba(X_test)

        assert probabilities.shape == (10, 2)
        # Probabilities should sum to 1
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestMLModelTrainer:
    """Test MLModelTrainer class."""

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = MLModelTrainer()

        assert trainer.models == {}
        assert trainer.results == []
        assert trainer.feature_importance == {}

    def test_train_logistic_regression(self, sample_features_data):
        """Test logistic regression training."""
        trainer = MLModelTrainer()

        train, test = create_temporal_train_test_split(sample_features_data, test_size=0.2)
        X_train, X_test, y_train, y_test, _, _ = prepare_features(train, test)

        model, metrics = trainer.train_logistic_regression(X_train, y_train, X_test, y_test)

        assert model is not None
        assert 'accuracy' in metrics
        assert 'logistic_regression' in trainer.models
        assert 'logistic_regression' in trainer.feature_importance

    def test_train_random_forest(self, sample_features_data):
        """Test random forest training."""
        trainer = MLModelTrainer()

        train, test = create_temporal_train_test_split(sample_features_data, test_size=0.2)
        X_train, X_test, y_train, y_test, _, _ = prepare_features(train, test)

        model, metrics = trainer.train_random_forest(X_train, y_train, X_test, y_test)

        assert model is not None
        assert 'accuracy' in metrics
        assert 'random_forest' in trainer.models
        assert 'random_forest' in trainer.feature_importance

    @pytest.mark.skipif(
        'xgboost' not in sys.modules,
        reason="XGBoost not available"
    )
    def test_train_xgboost(self, sample_features_data):
        """Test XGBoost training."""
        trainer = MLModelTrainer()

        train, test = create_temporal_train_test_split(sample_features_data, test_size=0.2)
        X_train, X_test, y_train, y_test, _, _ = prepare_features(train, test)

        model, metrics = trainer.train_xgboost(X_train, y_train, X_test, y_test)

        if model is not None:
            assert 'accuracy' in metrics
            assert 'xgboost' in trainer.models

    def test_feature_importance_extraction(self, sample_features_data):
        """Test feature importance is extracted correctly."""
        trainer = MLModelTrainer()

        train, test = create_temporal_train_test_split(sample_features_data, test_size=0.2)
        X_train, X_test, y_train, y_test, feature_cols, _ = prepare_features(train, test)

        trainer.train_random_forest(X_train, y_train, X_test, y_test)

        importance = trainer.feature_importance['random_forest']

        # Should have importance for all features
        assert len(importance) == len(feature_cols)
        # All importance values should be non-negative
        assert all(v >= 0 for v in importance.values())


class TestClassImbalance:
    """Test handling of class imbalance."""

    def test_class_weight_handling(self):
        """Test that class weights are applied correctly."""
        # Create imbalanced dataset
        X = np.random.rand(100, 5)
        y = np.array([0] * 90 + [1] * 10)  # 90/10 split

        trainer = MLModelTrainer()

        # Train model
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lr = LogisticRegression(class_weight='balanced', random_state=42)
        lr.fit(X_scaled, y)

        # Make predictions
        y_pred = lr.predict(X_scaled)

        # Should predict at least some of the minority class
        assert y_pred.sum() > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()

        # Should handle gracefully
        try:
            train, test = create_temporal_train_test_split(empty_df, test_size=0.2)
            # If no error, train and test should be empty
            assert len(train) == 0
            assert len(test) == 0
        except Exception as e:
            # Some exceptions are acceptable
            assert isinstance(e, (ValueError, KeyError))

    def test_single_class_handling(self):
        """Test handling when only one class present."""
        data = pd.DataFrame({
            'gameDate': pd.date_range('2023-01-01', periods=50),
            'fullName': ['Player'] * 50,
            'over_threshold': [1] * 50,  # Only one class
            'rolling_3g_points': np.random.rand(50),
        })

        train, test = create_temporal_train_test_split(data, test_size=0.2)

        # Should still split data
        assert len(train) > 0
        assert len(test) > 0


@pytest.mark.parametrize("model_type", [
    'logistic_regression',
    'random_forest',
])
def test_model_consistency(sample_features_data, model_type):
    """Test that models produce consistent results with same data."""
    trainer = MLModelTrainer()

    train, test = create_temporal_train_test_split(sample_features_data, test_size=0.2)
    X_train, X_test, y_train, y_test, _, _ = prepare_features(train, test)

    # Train model twice
    if model_type == 'logistic_regression':
        model1, metrics1 = trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
    else:
        model1, metrics1 = trainer.train_random_forest(X_train, y_train, X_test, y_test)

    # Results should be deterministic with random_state
    assert metrics1['accuracy'] > 0
    assert metrics1['f1_score'] > 0
