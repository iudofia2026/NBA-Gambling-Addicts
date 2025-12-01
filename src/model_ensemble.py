"""
Model Ensemble for Enhanced Prediction Accuracy
Combines multiple models for better prediction accuracy
Target: 75%+ accuracy through ensemble methods
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelEnsemble:
    """
    Ensemble of specialized models for NBA predictions
    Each model focuses on different aspects of the game
    """

    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.feature_importance = {}
        self.prediction_history = []
        self.accuracy_history = {}

        # Model initialization
        self._initialize_models()

    def _initialize_models(self):
        """Initialize different types of models with specializations"""

        # 1. Historical Pattern Model (XGBoost)
        # Focuses on long-term trends and patterns
        self.models['historical'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.model_weights['historical'] = 0.25

        # 2. Recent Form Model (Random Forest)
        # Focuses on last 10-15 games performance
        self.models['recent_form'] = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.model_weights['recent_form'] = 0.25

        # 3. Matchup Specialist (Logistic Regression)
        # Focuses on player vs team matchup history
        self.models['matchup'] = LogisticRegression(
            penalty='l2',
            C=1.0,
            random_state=42,
            max_iter=1000
        )
        self.model_weights['matchup'] = 0.20

        # 4. Context Model (Gradient Boosting)
        # Focuses on game context (rest, travel, importance)
        self.models['context'] = xgb.XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.15,
            random_state=42
        )
        self.model_weights['context'] = 0.15

        # 5. Market Intelligence Model (Random Forest)
        # Focuses on betting market signals
        self.models['market'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            random_state=42
        )
        self.model_weights['market'] = 0.15

    def train_ensemble(self, X, y, feature_groups=None):
        """
        Train each model on specialized features
        feature_groups: dict mapping model_name to list of feature indices
        """
        print("Training ensemble models...")

        if feature_groups is None:
            # Use all features for all models if not specified
            feature_groups = {name: list(range(X.shape[1])) for name in self.models.keys()}

        # Train each model
        for model_name, model in self.models.items():
            print(f"  Training {model_name} model...")

            # Select features for this model
            if model_name in feature_groups:
                X_subset = X.iloc[:, feature_groups[model_name]]
            else:
                X_subset = X

            # Train the model
            model.fit(X_subset, y)

            # Record feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = dict(zip(
                    X_subset.columns,
                    model.feature_importances_
                ))

        print("✅ All models trained successfully")

    def predict_proba_ensemble(self, X, feature_groups=None):
        """
        Make ensemble predictions using weighted voting
        Returns probability distribution over/under
        """
        predictions = {}

        if feature_groups is None:
            feature_groups = {name: list(range(X.shape[1])) for name in self.models.keys()}

        # Get predictions from each model
        for model_name, model in self.models.items():
            # Select features for this model
            if model_name in feature_groups:
                X_subset = X.iloc[:, feature_groups[model_name]]
            else:
                X_subset = X

            # Get probability predictions
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(X_subset)
                predictions[model_name] = pred_proba[:, 1]  # Probability of OVER
            else:
                # Fallback to predict and convert
                pred = model.predict(X_subset)
                predictions[model_name] = pred.astype(float)

        # Weighted ensemble
        ensemble_prediction = np.zeros(len(X))
        for model_name, pred in predictions.items():
            weight = self.model_weights[model_name]
            ensemble_prediction += pred * weight

        return ensemble_prediction

    def predict_ensemble(self, X, feature_groups=None):
        """
        Make binary predictions using ensemble
        Returns 1 for OVER, 0 for UNDER
        """
        proba = self.predict_proba_ensemble(X, feature_groups)
        return (proba > 0.5).astype(int)

    def update_weights(self, recent_performance):
        """
        Dynamically update model weights based on recent performance
        recent_performance: dict {model_name: accuracy}
        """
        # Normalize recent performance
        total_performance = sum(recent_performance.values())
        if total_performance > 0:
            for model_name in self.model_weights:
                # Blend old weight with new performance (70% old, 30% new)
                new_weight = recent_performance.get(model_name, 0) / total_performance
                self.model_weights[model_name] = (
                    0.7 * self.model_weights[model_name] + 0.3 * new_weight
                )

        # Normalize weights to sum to 1
        total_weight = sum(self.model_weights.values())
        for model_name in self.model_weights:
            self.model_weights[model_name] /= total_weight

    def save_ensemble(self, filepath):
        """Save the entire ensemble to disk"""
        ensemble_data = {
            'models': self.models,
            'weights': self.model_weights,
            'feature_importance': self.feature_importance,
            'prediction_history': self.prediction_history
        }
        joblib.dump(ensemble_data, filepath)
        print(f"Ensemble saved to {filepath}")

    def load_ensemble(self, filepath):
        """Load ensemble from disk"""
        ensemble_data = joblib.load(filepath)
        self.models = ensemble_data['models']
        self.model_weights = ensemble_data['weights']
        self.feature_importance = ensemble_data['feature_importance']
        self.prediction_history = ensemble_data.get('prediction_history', [])
        print(f"Ensemble loaded from {filepath}")

    def get_feature_importance_summary(self):
        """Get aggregated feature importance across all models"""
        all_importance = {}

        for model_name, importance in self.feature_importance.items():
            for feature, score in importance.items():
                if feature not in all_importance:
                    all_importance[feature] = []
                all_importance[feature].append(score)

        # Aggregate by mean
        aggregated_importance = {
            feature: np.mean(scores) for feature, scores in all_importance.items()
        }

        # Sort by importance
        sorted_importance = sorted(
            aggregated_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_importance

    def evaluate_ensemble(self, X_test, y_test, feature_groups=None):
        """
        Evaluate ensemble performance
        """
        # Get ensemble predictions
        y_pred_proba = self.predict_proba_ensemble(X_test, feature_groups)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)

        # Calculate ROC AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, y_pred_proba)

        # Calculate precision and recall
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Individual model performance
        individual_metrics = {}
        for model_name, model in self.models.items():
            if feature_groups and model_name in feature_groups:
                X_subset = X_test.iloc[:, feature_groups[model_name]]
            else:
                X_subset = X_test

            y_pred_model = model.predict(X_subset)
            individual_metrics[model_name] = {
                'accuracy': np.mean(y_pred_model == y_test),
                'precision': precision_score(y_test, y_pred_model),
                'recall': recall_score(y_test, y_pred_model)
            }

        return {
            'ensemble': {
                'accuracy': accuracy,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            'individual_models': individual_metrics
        }


class SpecializedFeatureSelector:
    """
    Selects optimal features for each model in the ensemble
    """

    def __init__(self):
        self.feature_groups = {}
        self.selected_features = {}

    def select_features_by_model(self, X, y):
        """
        Select best features for each model type
        """
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

        feature_names = X.columns.tolist()

        # For Historical model (XGBoost) - select top features
        selector_historical = SelectKBest(f_classif, k=30)
        selector_historical.fit(X, y)
        historical_features = [feature_names[i] for i in selector_historical.get_support(indices=True)]

        # For Recent Form model - focus on rolling averages and trends
        recent_keywords = ['rolling_', 'trend', 'streak', 'hot', 'cold']
        recent_features = [f for f in feature_names if any(kw in f.lower() for kw in recent_keywords)]

        # For Matchup model - focus on opponent and historical matchup data
        matchup_keywords = ['opponent', 'vs_', 'history', 'team_']
        matchup_features = [f for f in feature_names if any(kw in f.lower() for kw in matchup_keywords)]

        # For Context model - focus on rest, travel, game importance
        context_keywords = ['rest', 'travel', 'days_', 'home', 'away', 'importance', 'season']
        context_features = [f for f in feature_names if any(kw in f.lower() for kw in context_keywords)]

        # For Market model - focus on betting-related features (if available)
        market_keywords = ['line', 'odds', 'public', 'sharp', 'volume', 'movement']
        market_features = [f for f in feature_names if any(kw in f.lower() for kw in market_keywords)]

        self.feature_groups = {
            'historical': [feature_names.index(f) for f in historical_features],
            'recent_form': [feature_names.index(f) for f in recent_features],
            'matchup': [feature_names.index(f) for f in matchup_features],
            'context': [feature_names.index(f) for f in context_features],
            'market': [feature_names.index(f) for f in market_features]
        }

        # Ensure each model has at least some features
        for model_name in self.feature_groups:
            if len(self.feature_groups[model_name]) == 0:
                # Default to top 20 features for this model
                selector_default = SelectKBest(mutual_info_classif, k=20)
                selector_default.fit(X, y)
                self.feature_groups[model_name] = selector_default.get_support(indices=True).tolist()

        return self.feature_groups


# Integration function for your system
def create_enhanced_prediction_system():
    """
    Creates an enhanced prediction system using ensemble approach
    """
    # Load your data
    data = pd.read_csv('../data/processed/engineered_features.csv')

    # Prepare features (you would enhance this with new features)
    feature_cols = [col for col in data.columns if col not in [
        'over_threshold', 'gameDate', 'fullName', 'player_threshold',
        'points_vs_threshold'
    ]]

    X = data[feature_cols]
    y = data['over_threshold']

    # Initialize ensemble
    ensemble = ModelEnsemble()

    # Select features for each model
    feature_selector = SpecializedFeatureSelector()
    feature_groups = feature_selector.select_features_by_model(X, y)

    # Create temporal split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train ensemble
    ensemble.train_ensemble(X_train, y_train, feature_groups)

    # Evaluate
    metrics = ensemble.evaluate_ensemble(X_test, y_test, feature_groups)
    print("\nEnsemble Performance:")
    print(f"Accuracy: {metrics['ensemble']['accuracy']:.1%}")
    print(f"AUC: {metrics['ensemble']['auc']:.3f}")

    # Save ensemble
    ensemble.save_ensemble('../models/ensemble_model.pkl')

    return ensemble, feature_groups