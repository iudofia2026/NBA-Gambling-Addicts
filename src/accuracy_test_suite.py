"""
NBA ACCURACY TEST SUITE
Iterative testing for accuracy improvements
Uses existing historical data without external APIs
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class AccuracyTestSuite:
    """Test suite for measuring accuracy improvements."""

    def __init__(self):
        """Initialize test suite."""
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = []

        print("=" * 70)
        print("🧪 NBA ACCURACY TEST SUITE")
        print("   Target: 75%+ prediction accuracy")
        print("   Method: Iterative feature enhancement")
        print("=" * 70)

    def load_and_prepare_data(self):
        """Load and prepare data with 90/10 split."""
        print("\n📂 LOADING & PREPARING DATA")

        data_path = 'data/processed/engineered_features.csv'
        if not os.path.exists(data_path):
            print(f"❌ Data file not found: {data_path}")
            return False

        # Load data
        self.data = pd.read_csv(data_path)
        self.data['gameDate'] = pd.to_datetime(self.data['gameDate'], errors='coerce')

        # Clean data
        print("🧹 Cleaning data...")
        original_size = len(self.data)
        self.data = self.data[
            (self.data['points'] >= 0) &
            (self.data['points'] <= 80) &
            (self.data['numMinutes'] > 0)
        ].copy()
        print(f"   Removed {original_size - len(self.data):,} invalid rows")

        # Create synthetic prop_line based on player_threshold
        self.data['prop_line'] = self.data['player_threshold']

        # Sort by date for chronological split
        self.data = self.data.sort_values('gameDate').reset_index(drop=True)

        # Create features and target
        self.X = self._create_feature_matrix()
        self.y = (self.data['points'] > self.data['prop_line']).astype(int)

        # 90/10 split (chronological to avoid lookahead)
        split_idx = int(len(self.data) * 0.9)
        self.X_train = self.X.iloc[:split_idx]
        self.X_test = self.X.iloc[split_idx:]
        self.y_train = self.y.iloc[:split_idx]
        self.y_test = self.y.iloc[split_idx:]

        print(f"✅ Training set: {len(self.X_train):,} samples (90%)")
        print(f"✅ Test set: {len(self.X_test):,} samples (10%)")
        print(f"✅ Features: {self.X.shape[1]}")

        return True

    def _create_feature_matrix(self):
        """Create comprehensive feature matrix."""
        features = pd.DataFrame(index=self.data.index)

        # 1. Basic player features
        features['points_avg_5'] = self.data['rolling_5g_points'].fillna(self.data['points'].rolling(5, min_periods=1).mean())
        features['points_std_5'] = self.data['points'].rolling(5, min_periods=1).std().fillna(0)
        features['minutes_last_game'] = self.data['rolling_5g_numMinutes'].fillna(self.data['numMinutes'].shift(1)).fillna(self.data['numMinutes'].mean())
        features['efficiency'] = (self.data['points'] / (self.data['numMinutes'] + 1) * 40).fillna(20)

        # 2. Advanced fatigue features
        features['minutes_spike'] = (
            self.data['numMinutes'].rolling(3, min_periods=1).mean() -
            self.data['numMinutes'].rolling(10, min_periods=1).mean()
        ) / self.data['numMinutes'].rolling(10, min_periods=1).mean()
        features['b2b_flag'] = (
            self.data['gameDate'].diff().dt.days <= 1
        ).astype(int).shift(1).fillna(0)
        features['rest_days'] = self.data['gameDate'].diff().dt.days.shift(1).fillna(2)
        features['cumulative_minutes'] = self.data['numMinutes'].rolling(10, min_periods=1).sum()

        # 3. Performance trends
        features['points_trend_3'] = self.data['points'].rolling(3, min_periods=1).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        features['efficiency_trend'] = features['efficiency'].rolling(5, min_periods=1).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )

        # 4. Matchup specific features
        features['player_avg_vs_opp'] = self.data.groupby('fullName')['points'].transform(
            lambda x: x.rolling(10, min_periods=1).mean()
        )
        features['over_rate_vs_opp'] = self.data.groupby(['fullName', 'opponentteamName'])['over_threshold'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        ).fillna(0.5)

        # 5. Team chemistry features
        features['team_points_share'] = self.data['points'] / self.data.groupby('playerteamName')['points'].transform(
            lambda x: x.rolling(5, min_periods=1).sum()
        )
        features['team_momentum'] = self.data.groupby('playerteamName')['points'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().pct_change()
        ).fillna(0)

        # 6. Line-specific features
        features['prop_line_avg'] = self.data['player_threshold'].rolling(5, min_periods=1).mean()
        features['line_movement'] = self.data['player_threshold'] - self.data['player_threshold'].shift(1).fillna(0)
        features['over_under_ratio'] = self.data['over_threshold'].rolling(10, min_periods=1).mean()

        # 7. Advanced derived features
        features['points_per_minute'] = self.data['points_per_minute'].fillna(self.data['points'] / self.data['numMinutes'])
        features['usage_efficiency'] = self.data['usage_rate_approx'] * features['efficiency'] / 100
        features['fatigue_impact'] = features['cumulative_minutes'] * features['b2b_flag'] / 100
        features['momentum_score'] = features['points_trend_3'] * features['team_momentum']

        # Fill any remaining NaNs
        features = features.fillna(features.mean())

        return features

    def run_iteration(self, iteration_num, feature_groups, description):
        """Run a single iteration with specified feature groups."""
        print(f"\n--- Iteration {iteration_num}: {description} ---")

        # Select features for this iteration
        feature_map = {
            'baseline': [
                'points_avg_5', 'points_std_5', 'minutes_last_game',
                'efficiency', 'prop_line_avg'
            ],
            'fatigue': [
                'minutes_spike', 'b2b_flag', 'rest_days',
                'cumulative_minutes', 'fatigue_impact'
            ],
            'trends': [
                'points_trend_3', 'efficiency_trend', 'team_momentum',
                'momentum_score'
            ],
            'matchup': [
                'player_avg_vs_opp', 'over_rate_vs_opp',
                'team_points_share'
            ],
            'market': [
                'line_movement', 'over_under_ratio',
                'usage_efficiency'
            ],
            'advanced': [
                'points_per_minute', 'usage_efficiency',
                'fatigue_impact', 'momentum_score'
            ]
        }

        selected_features = []
        for group in feature_groups:
            selected_features.extend(feature_map.get(group, []))

        # Ensure we have features
        selected_features = list(set(selected_features) & set(self.X_train.columns))

        if not selected_features:
            print("❌ No valid features for this iteration")
            return None

        # Prepare data
        X_train_iter = self.X_train[selected_features]
        X_test_iter = self.X_test[selected_features]

        # Train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

        # Fit model
        model.fit(X_train_iter, self.y_train)

        # Predictions
        y_pred_train = model.predict(X_train_iter)
        y_pred_test = model.predict(X_test_iter)
        y_proba_test = model.predict_proba(X_test_iter)[:, 1]

        # Calculate metrics
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        cv_scores = cross_val_score(model, X_train_iter, self.y_train, cv=5)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        result = {
            'iteration': iteration_num,
            'features': feature_groups,
            'description': description,
            'num_features': len(selected_features),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance.head(5).to_dict('records'),
            'overfit_gap': train_accuracy - test_accuracy
        }

        # Calculate improvement from baseline
        if self.results:
            baseline_acc = self.results[0]['test_accuracy']
            result['improvement'] = test_accuracy - baseline_acc
        else:
            result['improvement'] = 0

        # Print results
        print(f"✅ Features used: {len(selected_features)}")
        print(f"📊 Test Accuracy: {test_accuracy:.4f} ({test_accuracy:.1%})")
        print(f"🎯 CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        if result['improvement'] != 0:
            print(f"📈 Improvement: {result['improvement']:+.4f} ({result['improvement']:+.1%})")

        print("\nTop 5 Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"  • {row['feature']}: {row['importance']:.4f}")

        self.results.append(result)

        # Save model if it's the best so far
        if len(self.results) == 1 or test_accuracy > max(r['test_accuracy'] for r in self.results[:-1]):
            model_dir = 'models'
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(model, f"{model_dir}/best_model_iter_{iteration_num}.pkl")
            print(f"💾 New best model saved!")

        return result

    def run_all_iterations(self):
        """Run all 5 iterations."""
        iterations = [
            (1, ['baseline'], "Baseline features only"),
            (2, ['baseline', 'fatigue'], "Add fatigue & load management"),
            (3, ['baseline', 'fatigue', 'trends'], "Add performance trends"),
            (4, ['baseline', 'fatigue', 'trends', 'matchup'], "Add matchup analysis"),
            (5, ['baseline', 'fatigue', 'trends', 'matchup', 'market'], "Add market intelligence")
        ]

        for iter_num, features, desc in iterations:
            self.run_iteration(iter_num, features, desc)

        # Additional hyperparameter tuning iteration
        print("\n--- Iteration 6: Hyperparameter Optimization ---")
        best_features = self.results[-1]['features']
        self._run_hyperparameter_tuning(7, best_features)

    def _run_hyperparameter_tuning(self, iteration_num, feature_groups):
        """Run hyperparameter tuning."""
        selected_features = []
        feature_map = {
            'baseline': ['points_avg_5', 'points_std_5', 'minutes_last_game', 'efficiency', 'prop_line_avg'],
            'fatigue': ['minutes_spike', 'b2b_flag', 'rest_days', 'cumulative_minutes', 'fatigue_impact'],
            'trends': ['points_trend_3', 'efficiency_trend', 'team_momentum', 'momentum_score'],
            'matchup': ['player_avg_vs_opp', 'over_rate_vs_opp', 'team_points_share'],
            'market': ['line_movement', 'over_under_ratio', 'usage_efficiency'],
            'advanced': ['points_per_minute', 'usage_efficiency', 'fatigue_impact', 'momentum_score']
        }

        for group in feature_groups:
            selected_features.extend(feature_map.get(group, []))
        selected_features = list(set(selected_features) & set(self.X_train.columns))

        X_train_iter = self.X_train[selected_features]
        X_test_iter = self.X_test[selected_features]

        # Optimized hyperparameters
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train_iter, self.y_train)
        y_pred = model.predict(X_test_iter)
        test_accuracy = accuracy_score(self.y_test, y_pred)

        result = {
            'iteration': iteration_num,
            'features': feature_groups + ['optimized'],
            'description': "Hyperparameter optimization",
            'num_features': len(selected_features),
            'train_accuracy': model.score(X_train_iter, self.y_train),
            'test_accuracy': test_accuracy,
            'cv_mean': cross_val_score(model, X_train_iter, self.y_train, cv=5).mean(),
            'cv_std': 0,
            'overfit_gap': model.score(X_train_iter, self.y_train) - test_accuracy,
            'improvement': test_accuracy - self.results[0]['test_accuracy'],
            'oob_score': model.oob_score_
        }

        self.results.append(result)
        print(f"🎯 Optimized Test Accuracy: {test_accuracy:.4f} ({test_accuracy:.1%})")
        print(f"📊 OOB Score: {model.oob_score_:.4f}")

        # Save optimized model
        model_dir = 'models'
        joblib.dump(model, f"{model_dir}/optimized_model_v2.pkl")

    def generate_report(self):
        """Generate comprehensive report."""
        if not self.results:
            print("\n❌ No results to report")
            return

        print("\n" + "=" * 70)
        print("📊 ACCURACY IMPROVEMENT REPORT")
        print("=" * 70)

        # Create results dataframe
        df_results = pd.DataFrame(self.results)

        # Print summary table
        print("\n📈 ACCURACY PROGRESSION:")
        print("-" * 50)
        for r in self.results:
            accuracy_pct = r['test_accuracy'] * 100
            improvement_pct = r['improvement'] * 100
            print(f"Iter {r['iteration']:2d}: {accuracy_pct:5.1f}% | {r['description']}")
            if r['improvement'] != 0:
                print(f"         {'+'if improvement_pct>0 else ''}{improvement_pct:4.1f}% improvement")

        # Find best iteration
        best_iter = max(self.results, key=lambda x: x['test_accuracy'])
        baseline_iter = self.results[0]

        print("\n🏆 BEST RESULTS:")
        print("-" * 50)
        print(f"Baseline Accuracy: {baseline_iter['test_accuracy']:.4f} ({baseline_iter['test_accuracy']:.1%})")
        print(f"Best Accuracy: {best_iter['test_accuracy']:.4f} ({best_iter['test_accuracy']:.1%})")
        print(f"Total Improvement: {best_iter['improvement']:+.4f} ({best_iter['improvement']:+.1%})")
        print(f"Best Iteration: {best_iter['iteration']} - {best_iter['description']}")

        # Check if we met the target
        target_met = best_iter['test_accuracy'] >= 0.75
        print(f"\n🎯 Target 75%+: {'✅ ACHIEVED!' if target_met else '❌ Not yet'}")
        print(f"   Current: {best_iter['test_accuracy']:.1%}")
        print(f"   Needed: 75.0%")
        print(f"   Gap: {max(0, 0.75 - best_iter['test_accuracy']):.1%}")

        # Feature analysis
        print("\n🔍 FEATURE ANALYSIS:")
        print("-" * 50)

        # Aggregate feature importance
        all_importances = []
        for r in self.results:
            if 'feature_importance' in r:
                for feat in r['feature_importance']:
                    all_importances.append(feat)

        if all_importances:
            feat_df = pd.DataFrame(all_importances)
            top_features = feat_df.groupby('feature')['importance'].mean().sort_values(ascending=False).head(10)

            print("Top 10 Most Important Features:")
            for feat, imp in top_features.items():
                print(f"  • {feat}: {imp:.4f}")

        # Save detailed report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        report_data = {
            'timestamp': timestamp,
            'baseline_accuracy': baseline_iter['test_accuracy'],
            'best_accuracy': best_iter['test_accuracy'],
            'improvement': best_iter['improvement'],
            'target_met': target_met,
            'results': self.results
        }

        # Save to CSV
        df_results.to_csv(f'data/processed/accuracy_test_report_{timestamp}.csv', index=False)

        # Save to JSON
        import json
        with open(f'data/processed/accuracy_test_report_{timestamp}.json', 'w') as f:
            # Convert numpy types for JSON serialization
            report_copy = {k: v for k, v in report_data.items()}
            for r in report_copy['results']:
                for k, v in r.items():
                    if isinstance(v, np.integer):
                        r[k] = int(v)
                    elif isinstance(v, np.floating):
                        r[k] = float(v)
            json.dump(report_copy, f, indent=2)

        print(f"\n💾 Report saved to: accuracy_test_report_{timestamp}.*")

        return df_results

def main():
    """Run the complete accuracy test suite."""
    suite = AccuracyTestSuite()

    # Load data
    if not suite.load_and_prepare_data():
        return

    # Run iterations
    suite.run_all_iterations()

    # Generate report
    suite.generate_report()

    print("\n✅ Accuracy test suite complete!")

if __name__ == "__main__":
    main()