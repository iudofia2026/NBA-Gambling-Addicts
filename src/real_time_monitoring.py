"""
Real-time Monitoring System for NBA Betting Model
Tracks accuracy, model performance, and prediction quality
Provides alerts and insights for continuous improvement
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import json
import warnings
warnings.filterwarnings('ignore')

class ModelMonitor:
    """
    Monitors model performance in real-time
    Tracks accuracy trends and provides insights
    """

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.model_outputs = {}  # Store outputs from each model
        self.feature_importance_history = []
        self.alerts = []

        # Performance metrics
        self.current_accuracy = 0
        self.rolling_accuracy = deque(maxlen=30)
        self.confidence_calibration = []

        # Target tracking
        self.target_accuracy = 0.75
        self.accuracy_milestones = [0.65, 0.70, 0.75, 0.80]

    def log_prediction(self, prediction, actual, confidence, model_outputs=None, features=None):
        """
        Log a prediction and its actual outcome
        """
        timestamp = datetime.now()

        # Store basic prediction data
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.confidences.append(confidence)

        # Store model outputs if provided
        if model_outputs:
            for model_name, output in model_outputs.items():
                if model_name not in self.model_outputs:
                    self.model_outputs[model_name] = deque(maxlen=self.window_size)
                self.model_outputs[model_name].append({
                    'output': output,
                    'timestamp': timestamp
                })

        # Calculate metrics
        self._update_metrics()

        # Check for alerts
        self._check_alerts()

    def _update_metrics(self):
        """Update performance metrics"""
        if len(self.predictions) < 5:
            return

        # Current accuracy
        predictions_array = np.array(list(self.predictions))
        actuals_array = np.array(list(self.actuals))
        self.current_accuracy = np.mean(predictions_array == actuals_array)

        # Rolling accuracy (last 30 predictions)
        if len(predictions_array) >= 30:
            rolling_pred = predictions_array[-30:]
            rolling_act = actuals_array[-30:]
            rolling_acc = np.mean(rolling_pred == rolling_act)
            self.rolling_accuracy.append(rolling_acc)

        # Confidence calibration
        self._update_confidence_calibration()

    def _update_confidence_calibration(self):
        """Check if confidence scores are well-calibrated"""
        if len(self.predictions) < 10:
            return

        confidences = np.array(list(self.confidences))
        correct = np.array(list(self.predictions)) == np.array(list(self.actuals))

        # Group by confidence bins
        bins = np.linspace(0.5, 1, 10)
        calibration_data = []

        for i in range(len(bins) - 1):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if np.sum(mask) > 0:
                bin_confidence = np.mean(confidences[mask])
                bin_accuracy = np.mean(correct[mask])
                calibration_data.append({
                    'confidence_bin': bin_confidence,
                    'actual_accuracy': bin_accuracy,
                    'count': np.sum(mask)
                })

        self.confidence_calibration = calibration_data

    def _check_alerts(self):
        """Check for performance alerts"""
        # Accuracy drop alert
        if len(self.rolling_accuracy) >= 5:
            recent_avg = np.mean(list(self.rolling_accuracy)[-5:])
            if recent_avg < 0.55:  # Significant drop
                self.add_alert('accuracy_drop', f"Accuracy dropped to {recent_avg:.1%}")

        # Confidence mismatch alert
        if self.confidence_calibration:
            max_mismatch = max(
                abs(c['confidence_bin'] - c['actual_accuracy'])
                for c in self.confidence_calibration
            )
            if max_mismatch > 0.2:
                self.add_alert('confidence_mismatch', f"Max confidence mismatch: {max_mismatch:.1%}")

        # Milestone alert
        for milestone in self.accuracy_milestones:
            if self.current_accuracy >= milestone and not hasattr(self, f'alerted_{milestone}'):
                self.add_alert('milestone', f"Reached {milestone:.0%} accuracy!")
                setattr(self, f'alerted_{milestone}', True)

    def add_alert(self, alert_type, message):
        """Add an alert"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now(),
            'accuracy': self.current_accuracy
        }
        self.alerts.append(alert)
        print(f"🚨 ALERT [{alert_type.upper()}]: {message}")

    def get_performance_report(self):
        """Generate comprehensive performance report"""
        report = {
            'current_accuracy': self.current_accuracy,
            'target_accuracy': self.target_accuracy,
            'progress_to_target': self.current_accuracy / self.target_accuracy,
            'total_predictions': len(self.predictions),
            'recent_trend': self._calculate_trend(),
            'confidence_calibration': self.confidence_calibration,
            'alerts': self.alerts[-10:],  # Last 10 alerts
            'model_performance': self._get_model_performance()
        }

        return report

    def _calculate_trend(self):
        """Calculate recent accuracy trend"""
        if len(self.rolling_accuracy) < 10:
            return "insufficient_data"

        recent = list(self.rolling_accuracy)[-10:]
        if len(recent) >= 5:
            slope = np.polyfit(range(len(recent)), recent, 1)[0]
            if slope > 0.01:
                return "improving"
            elif slope < -0.01:
                return "declining"
            else:
                return "stable"
        return "insufficient_data"

    def _get_model_performance(self):
        """Get performance metrics for each model"""
        model_performance = {}

        for model_name, outputs in self.model_outputs.items():
            if len(outputs) < 5:
                continue

            # Calculate model-specific accuracy
            correct_count = 0
            for output in outputs:
                # This would need actual implementation based on model outputs
                # Placeholder for now
                pass

            model_performance[model_name] = {
                'predictions': len(outputs),
                # Add model-specific metrics here
            }

        return model_performance

    def save_performance_history(self, filepath):
        """Save performance history to file"""
        history = {
            'predictions': list(self.predictions),
            'actuals': list(self.actuals),
            'confidences': list(self.confidences),
            'rolling_accuracy': list(self.rolling_accuracy),
            'accuracy_history': [self.current_accuracy],
            'alerts': self.alerts,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)

    def plot_performance_dashboard(self, save_path=None):
        """Create performance dashboard visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('NBA Betting Model Performance Dashboard', fontsize=16)

        # 1. Accuracy over time
        if len(self.rolling_accuracy) > 0:
            axes[0, 0].plot(list(self.rolling_accuracy), marker='o')
            axes[0, 0].axhline(y=self.target_accuracy, color='r', linestyle='--', label=f'Target: {self.target_accuracy:.0%}')
            axes[0, 0].set_title('Rolling Accuracy (Last 30 predictions)')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

        # 2. Confidence calibration
        if self.confidence_calibration:
            conf_bins = [c['confidence_bin'] for c in self.confidence_calibration]
            act_acc = [c['actual_accuracy'] for c in self.confidence_calibration]
            axes[0, 1].scatter(conf_bins, act_acc, alpha=0.7)
            axes[0, 1].plot([0.5, 1], [0.5, 1], 'r--', label='Perfect Calibration')
            axes[0, 1].set_title('Confidence Calibration')
            axes[0, 1].set_xlabel('Predicted Probability')
            axes[0, 1].set_ylabel('Actual Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        # 3. Prediction distribution
        if self.predictions:
            predictions_array = np.array(list(self.predictions))
            actuals_array = np.array(list(self.actuals))
            axes[1, 0].hist(predictions_array, bins=20, alpha=0.7, label='Predictions')
            axes[1, 0].hist(actuals_array, bins=20, alpha=0.7, label='Actual')
            axes[1, 0].set_title('Prediction vs Actual Distribution')
            axes[1, 0].legend()
            axes[1, 0].set_xlabel('Value')
            axes[1, 0].set_ylabel('Frequency')

        # 4. Performance metrics table
        metrics_text = f"""
        Current Metrics:
        ─────────────────
        Accuracy: {self.current_accuracy:.1%}
        Target: {self.target_accuracy:.1%}
        Progress: {self.current_accuracy/self.target_accuracy:.1%}
        Total Predictions: {len(self.predictions)}
        Trend: {self._calculate_trend()}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                        fontfamily='monospace')
        axes[1, 1].set_title('Performance Summary')
        axes[1, 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


class AccuracyTracker:
    """
    Tracks progress toward 75% accuracy goal
    """

    def __init__(self):
        self.accuracy_history = []
        self.improvement_log = []
        self.feature_impacts = {}
        self.milestone_progress = {}

    def log_daily_accuracy(self, date, accuracy, features_added=None):
        """Log daily accuracy and any changes"""
        self.accuracy_history.append({
            'date': date,
            'accuracy': accuracy,
            'features_added': features_added or []
        })

        # Calculate improvement from previous day
        if len(self.accuracy_history) > 1:
            prev_accuracy = self.accuracy_history[-2]['accuracy']
            improvement = accuracy - prev_accuracy
            self.improvement_log.append({
                'date': date,
                'improvement': improvement,
                'features_added': features_added or []
            })

    def analyze_feature_impact(self):
        """Analyze which features contributed most to accuracy gains"""
        feature_scores = {}
        for log in self.improvement_log:
            if log['improvement'] > 0.01:  # Significant improvement
                for feature in log['features_added']:
                    if feature not in feature_scores:
                        feature_scores[feature] = []
                    feature_scores[feature].append(log['improvement'])

        # Calculate average impact
        self.feature_impacts = {
            feature: np.mean(scores) for feature, scores in feature_scores.items()
        }

        return sorted(self.feature_impacts.items(), key=lambda x: x[1], reverse=True)

    def get_progress_to_goal(self):
        """Calculate progress to 75% accuracy goal"""
        if not self.accuracy_history:
            return 0

        current_accuracy = self.accuracy_history[-1]['accuracy']
        return min(current_accuracy / 0.75, 1.0)

    def predict_time_to_goal(self):
        """Predict time to reach 75% based on recent improvement rate"""
        if len(self.accuracy_history) < 7:
            return "Insufficient data"

        recent_improvements = [log['improvement'] for log in self.improvement_log[-7:]]
        avg_daily_improvement = np.mean(recent_improvements)

        if avg_daily_improvement <= 0:
            return "Not improving"

        current_accuracy = self.accuracy_history[-1]['accuracy']
        needed_improvement = 0.75 - current_accuracy
        days_to_goal = needed_improvement / avg_daily_improvement

        return f"{days_to_goal:.0f} days at current rate"


# Integration function
def setup_monitoring_system():
    """
    Set up monitoring for your prediction system
    """
    monitor = ModelMonitor(window_size=200)
    tracker = AccuracyTracker()

    return monitor, tracker


# Example usage in your prediction system
def integrate_monitoring_with_predictions():
    """
    Example of how to integrate monitoring with final_predictions_optimized.py
    """
    # Initialize monitoring
    monitor, tracker = setup_monitoring_system()

    # In your prediction loop:
    def make_and_track_prediction(player_name, prop_line, actual_result=None):
        """
        Make prediction and track its outcome
        """
        # Make your prediction here
        prediction = your_model.predict(player_name, prop_line)
        confidence = your_model.get_confidence()

        # Log the prediction
        if actual_result is not None:
            monitor.log_prediction(
                prediction=prediction,
                actual=actual_result,
                confidence=confidence,
                model_outputs={
                    'historical': model_historical_output,
                    'recent_form': model_recent_output,
                    'matchup': model_matchup_output
                }
            )

            # Track daily accuracy
            tracker.log_daily_accuracy(
                date=datetime.now().date(),
                accuracy=monitor.current_accuracy,
                features_added=['fatigue_metrics', 'opponent_defense']
            )

        return prediction

    return make_and_track_prediction