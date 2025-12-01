"""
NBA Baseline Models

This module implements simple baseline models for over/under predictions to establish
performance benchmarks before training more complex ML models.

Baselines:
1. Season Average vs Threshold
2. Rolling 5-Game Average vs Threshold
3. Rolling 10-Game Average vs Threshold
4. Random Baseline (50/50)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_engineered_data():
    """Load the engineered features dataset."""

    print("=== NBA BASELINE MODELS ===")
    print("Loading engineered features dataset...")

    # Load engineered features
    data = pd.read_csv('../data/processed/engineered_features.csv')

    # Convert gameDate to datetime
    data['gameDate'] = pd.to_datetime(data['gameDate'], errors='coerce')

    print(f"✓ Loaded {len(data):,} games with {data.shape[1]} features")
    print(f"✓ Players: {data['fullName'].nunique()}")
    print(f"✓ Date range: {data['gameDate'].min()} to {data['gameDate'].max()}")
    print(f"✓ Over rate: {data['over_threshold'].mean():.1%}")

    return data

def create_temporal_splits(data):
    """Create temporal train/test splits to prevent look-ahead bias."""

    print("\nCreating temporal train/test splits...")

    # Convert gameDate to datetime
    data['gameDate'] = pd.to_datetime(data['gameDate'])

    # Sort by date
    data = data.sort_values('gameDate').copy()

    # Use last 20% of data for testing (most recent games)
    cutoff_idx = int(len(data) * 0.8)
    cutoff_date = data.iloc[cutoff_idx]['gameDate']

    train_data = data[data['gameDate'] < cutoff_date].copy()
    test_data = data[data['gameDate'] >= cutoff_date].copy()

    print(f"✓ Train set: {len(train_data):,} games ({train_data['gameDate'].min()} to {train_data['gameDate'].max()})")
    print(f"✓ Test set: {len(test_data):,} games ({test_data['gameDate'].min()} to {test_data['gameDate'].max()})")
    print(f"✓ Train over rate: {train_data['over_threshold'].mean():.1%}")
    print(f"✓ Test over rate: {test_data['over_threshold'].mean():.1%}")

    return train_data, test_data

def evaluate_predictions(y_true, y_pred, y_proba, model_name):
    """Calculate comprehensive evaluation metrics."""

    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'log_loss': log_loss(y_true, y_proba) if y_proba is not None else None
    }

    return metrics

class BaselineModel:
    """Base class for baseline models."""

    def __init__(self, name):
        self.name = name
        self.predictions = None
        self.probabilities = None

    def fit(self, train_data):
        """Fit the baseline model (most baselines don't require training)."""
        pass

    def predict(self, test_data):
        """Make predictions on test data."""
        raise NotImplementedError

    def predict_proba(self, test_data):
        """Return prediction probabilities."""
        raise NotImplementedError

class SeasonAverageBaseline(BaselineModel):
    """Baseline using season average vs threshold."""

    def __init__(self):
        super().__init__("Season Average")
        self.player_averages = {}

    def fit(self, train_data):
        """Calculate season averages for each player."""
        self.player_averages = train_data.groupby('fullName')['points'].mean().to_dict()

    def predict(self, test_data):
        """Predict based on season average vs threshold."""
        test_data['predicted_points'] = test_data['fullName'].map(self.player_averages)
        predictions = (test_data['predicted_points'] > test_data['player_threshold']).astype(int)
        return predictions

    def predict_proba(self, test_data):
        """Return probability based on how far above/below threshold."""
        test_data['predicted_points'] = test_data['fullName'].map(self.player_averages)
        diff = test_data['predicted_points'] - test_data['player_threshold']

        # Convert difference to probability using sigmoid-like function
        probabilities = 1 / (1 + np.exp(-diff))
        return probabilities

class Rolling5GameBaseline(BaselineModel):
    """Baseline using 5-game rolling average vs threshold."""

    def __init__(self):
        super().__init__("Rolling 5-Game Average")

    def predict(self, test_data):
        """Predict based on 5-game rolling average."""
        predictions = (test_data['rolling_5g_points'] > test_data['player_threshold']).astype(int)
        return predictions

    def predict_proba(self, test_data):
        """Return probability based on 5-game average vs threshold."""
        diff = test_data['rolling_5g_points'] - test_data['player_threshold']
        probabilities = 1 / (1 + np.exp(-diff))
        return probabilities

class Rolling10GameBaseline(BaselineModel):
    """Baseline using 10-game rolling average vs threshold."""

    def __init__(self):
        super().__init__("Rolling 10-Game Average")

    def predict(self, test_data):
        """Predict based on 10-game rolling average."""
        predictions = (test_data['rolling_10g_points'] > test_data['player_threshold']).astype(int)
        return predictions

    def predict_proba(self, test_data):
        """Return probability based on 10-game average vs threshold."""
        diff = test_data['rolling_10g_points'] - test_data['player_threshold']
        probabilities = 1 / (1 + np.exp(-diff))
        return probabilities

class TrendBaseline(BaselineModel):
    """Baseline using recent trend and form."""

    def __init__(self):
        super().__init__("Trend-Based")

    def predict(self, test_data):
        """Predict based on recent trend vs threshold."""
        # Use rolling 5-game average adjusted by recent trend
        trend_adjusted = (
            test_data['rolling_5g_points'] +
            test_data['vs_threshold_trend_3g'].fillna(0)
        )
        predictions = (trend_adjusted > test_data['player_threshold']).astype(int)
        return predictions

    def predict_proba(self, test_data):
        """Return probability based on trend-adjusted average."""
        trend_adjusted = (
            test_data['rolling_5g_points'] +
            test_data['vs_threshold_trend_3g'].fillna(0)
        )
        diff = trend_adjusted - test_data['player_threshold']
        probabilities = 1 / (1 + np.exp(-diff))
        return probabilities

class RandomBaseline(BaselineModel):
    """Random baseline for comparison."""

    def __init__(self):
        super().__init__("Random")

    def predict(self, test_data):
        """Random predictions."""
        np.random.seed(42)  # For reproducibility
        return np.random.randint(0, 2, size=len(test_data))

    def predict_proba(self, test_data):
        """Random probabilities."""
        np.random.seed(42)
        return np.random.random(size=len(test_data))

def run_baseline_evaluation():
    """Run all baseline models and evaluate performance."""

    print("\nRunning baseline model evaluation...")

    # Load data
    data = load_engineered_data()

    # Create temporal splits
    train_data, test_data = create_temporal_splits(data)

    # Initialize baseline models
    models = [
        SeasonAverageBaseline(),
        Rolling5GameBaseline(),
        Rolling10GameBaseline(),
        TrendBaseline(),
        RandomBaseline()
    ]

    # Evaluate each model
    results = []

    for model in models:
        print(f"\n--- Evaluating {model.name} ---")

        # Fit model if needed
        model.fit(train_data)

        # Make predictions
        y_true = test_data['over_threshold']
        y_pred = model.predict(test_data)
        y_proba = model.predict_proba(test_data)

        # Handle missing values in predictions
        valid_mask = ~(pd.isna(y_pred) | pd.isna(y_proba))
        if valid_mask.sum() < len(y_pred):
            print(f"  Warning: {(~valid_mask).sum()} invalid predictions")

        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        y_proba_valid = y_proba[valid_mask] if y_proba is not None else None

        # Calculate metrics
        metrics = evaluate_predictions(y_true_valid, y_pred_valid, y_proba_valid, model.name)

        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")
        print(f"  Log Loss: {metrics['log_loss']:.3f}" if metrics['log_loss'] else "  Log Loss: N/A")

        results.append(metrics)

    return pd.DataFrame(results)

def player_specific_analysis(data, train_data, test_data):
    """Analyze baseline performance by individual player."""

    print("\nPlayer-specific baseline analysis...")

    # Use Rolling 5-game baseline for player analysis
    model = Rolling5GameBaseline()
    model.fit(train_data)

    y_pred = model.predict(test_data)
    valid_mask = ~pd.isna(y_pred)

    test_with_pred = test_data[valid_mask].copy()
    test_with_pred['baseline_pred'] = y_pred[valid_mask]

    # Calculate per-player metrics
    player_results = []

    for player in test_with_pred['fullName'].unique():
        player_data = test_with_pred[test_with_pred['fullName'] == player]

        if len(player_data) > 0:
            accuracy = accuracy_score(player_data['over_threshold'], player_data['baseline_pred'])
            f1 = f1_score(player_data['over_threshold'], player_data['baseline_pred'])

            player_results.append({
                'player': player,
                'test_games': len(player_data),
                'accuracy': accuracy,
                'f1_score': f1,
                'actual_over_rate': player_data['over_threshold'].mean(),
                'avg_threshold': player_data['player_threshold'].mean()
            })

    player_df = pd.DataFrame(player_results).sort_values('accuracy', ascending=False)

    print("Top performing players (5-game rolling baseline):")
    print(player_df.head())

    return player_df

def save_baseline_results(results, player_results):
    """Save baseline evaluation results."""

    print("\nSaving baseline results...")

    # Ensure directory exists
    os.makedirs('../data/processed', exist_ok=True)

    # Save overall results
    results_file = '../data/processed/baseline_results.csv'
    results.to_csv(results_file, index=False)

    # Save player-specific results
    player_file = '../data/processed/baseline_player_results.csv'
    player_results.to_csv(player_file, index=False)

    print(f"✓ Saved baseline results: {results_file}")
    print(f"✓ Saved player results: {player_file}")

    # Create summary
    best_model = results.loc[results['accuracy'].idxmax()]
    summary = {
        'best_baseline_model': best_model['model'],
        'best_accuracy': best_model['accuracy'],
        'best_f1_score': best_model['f1_score'],
        'random_accuracy': results[results['model'] == 'Random']['accuracy'].values[0],
        'evaluation_timestamp': datetime.now().isoformat(),
        'avg_player_accuracy': player_results['accuracy'].mean(),
        'best_player_accuracy': player_results['accuracy'].max(),
        'worst_player_accuracy': player_results['accuracy'].min()
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('../data/processed/baseline_summary.csv', index=False)

    return summary

def main():
    """Main baseline evaluation workflow."""

    # Load and prepare data
    data = load_engineered_data()

    # Create temporal splits
    train_data, test_data = create_temporal_splits(data)

    # Run baseline evaluation
    results = run_baseline_evaluation()

    # Player-specific analysis
    player_results = player_specific_analysis(data, train_data, test_data)

    # Save results
    summary = save_baseline_results(results, player_results)

    print("\n=== BASELINE EVALUATION COMPLETE ===")
    print(f"✅ Best baseline: {summary['best_baseline_model']}")
    print(f"✅ Best accuracy: {summary['best_accuracy']:.3f}")
    print(f"✅ Best F1 score: {summary['best_f1_score']:.3f}")
    print(f"✅ Beat random by: {summary['best_accuracy'] - summary['random_accuracy']:.3f}")
    print(f"✅ Average player accuracy: {summary['avg_player_accuracy']:.3f}")
    print(f"✅ Ready for ML model training!")

    return results, player_results, summary

if __name__ == "__main__":
    main()