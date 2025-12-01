"""
Enhanced NBA Predictions - Iteration 2
Integration of shot quality, team dynamics, and advanced analytics
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import modules
from odds_api_client import NBAOddsClient
from scaled_lr import ScaledLogisticRegression
from advanced_features_v1 import AdvancedFeatureGeneratorV1
from external_factors_v1 import ExternalFactorsV1
from shot_quality_v2 import ShotQualityFeaturesV2

class EnhancedPredictorV2:
    """Enhanced predictor with Iteration 1 & 2 features."""

    def __init__(self, api_key=None):
        """Initialize with API key."""
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError("API key required")

        self.odds_client = NBAOddsClient(self.api_key)
        self.models = {}
        self.feature_cols = None
        self.label_encoders = {}
        self.historical_data = None

        # Initialize feature generators
        self.advanced_generator = None
        self.external_generator = None
        self.shot_generator = None

    def load_models_and_data(self):
        """Load models and initialize all feature generators."""
        print("\n=== LOADING MODELS AND DATA (ITERATION 2) ===")

        # Load models
        model_dir = '../models'
        model_files = {
            'random_forest': 'random_forest_model.pkl',
            'xgboost': 'xgboost_model.pkl'
        }

        for model_name, filename in model_files.items():
            filepath = f"{model_dir}/{filename}"
            if os.path.exists(filepath):
                self.models[model_name] = joblib.load(filepath)
                print(f"✓ Loaded {model_name} model")

        # Load feature columns
        feature_file = f"{model_dir}/feature_columns.pkl"
        if os.path.exists(feature_file):
            self.feature_cols = joblib.load(feature_file)
            print(f"✓ Loaded {len(self.feature_cols)} feature columns")

        # Load historical data
        self.historical_data = pd.read_csv('../data/processed/engineered_features.csv')
        self.historical_data['gameDate'] = pd.to_datetime(self.historical_data['gameDate'], errors='coerce')

        print(f"✓ Loaded {len(self.historical_data):,} historical games")

        # Initialize all feature generators
        self.advanced_generator = AdvancedFeatureGeneratorV1(self.historical_data)
        self.external_generator = ExternalFactorsV1(self.historical_data)
        self.shot_generator = ShotQualityFeaturesV2(self.historical_data)

        print("✅ All feature generators initialized (v1 + v2)")

    def prepare_v2_features(self, player_name, game_context):
        """Prepare all features from iterations 1 & 2."""

        features = {}

        # Get baseline game features
        player_data = self.historical_data[self.historical_data['fullName'] == player_name]
        if player_data.empty:
            return None

        baseline = player_data.sort_values('gameDate').iloc[-1].copy()
        baseline['gameDate'] = pd.to_datetime(game_context['game_date'])
        baseline['year'] = pd.to_datetime(game_context['game_date']).year

        # Update team context
        baseline['playerteamName'] = game_context.get('player_team', baseline.get('playerteamName', ''))
        baseline['opponentteamName'] = game_context['opponent_team']

        # Generate Iteration 1 features
        advanced_features = self.advanced_generator.generate_all_advanced_features(
            player_name, game_context['opponent_team']
        )
        for feature, value in advanced_features.items():
            if feature in self.feature_cols:
                baseline[feature] = value
            else:
                self.feature_cols = list(self.feature_cols) + [feature]
                baseline[feature] = value

        # Generate external factors (Iteration 1)
        external_features = self.external_generator.generate_all_external_features(
            player_name, game_context['game_date'],
            game_context.get('player_team', ''),
            game_context['opponent_team']
        )
        for feature, value in external_features.items():
            if feature in self.feature_cols:
                baseline[feature] = value
            else:
                self.feature_cols = list(self.feature_cols) + [feature]
                baseline[feature] = value

        # Generate Iteration 2 shot quality features
        shot_features = self.shot_generator.generate_all_shot_quality_features(
            player_name, game_context.get('player_team', '')
        )
        for feature, value in shot_features.items():
            if feature in self.feature_cols:
                baseline[feature] = value
            else:
                self.feature_cols = list(self.feature_cols) + [feature]
                baseline[feature] = value

        # Composite scoring prediction using all iterations
        v1_prediction = advanced_features.get('scoring_expectation', baseline.get('points', 0))
        v1_adjustment = external_features.get('external_scoring_adjustment', 0)
        v2_potential = shot_features.get('adjusted_scoring_potential', v1_prediction)
        v2_situational = shot_features.get('game_situation_multiplier', 1.0)

        # Weighted combination (v2 gets more weight for shot quality)
        baseline['comprehensive_scoring_prediction'] = (
            v1_prediction * 0.3 +      # Weighted performance (v1)
            v2_potential * 0.5 +        # Shot quality and usage (v2)
            v1_adjustment * 0.2        # External factors (v1)
        ) * v2_situational

        # Quality confidence score
        baseline['prediction_confidence'] = min(
            (shot_features.get('shot_shot_quality_score', 0.5) * 0.4 +
             shot_features.get('clutch_clutch_consistency', 0.5) * 0.3 +
             advanced_features.get('matchup_vs_opp_experience', 0.5) * 0.3),
            1.0
        )

        return baseline

    def make_v2_prediction(self, player_name, prop_line, market_type, game_context):
        """Make prediction with all v1 + v2 features."""

        try:
            # Prepare features
            features = self.prepare_v2_features(player_name, game_context)
            if features is None:
                return None

            # Prepare feature vector
            feature_vector = features[self.feature_cols].to_frame().T

            # Handle categorical features
            for col, encoder in self.label_encoders.items():
                if col in feature_vector.columns:
                    try:
                        if feature_vector[col].iloc[0] not in encoder.classes_:
                            feature_vector[col] = 'unknown'
                        feature_vector[col] = encoder.transform(feature_vector[col].astype(str))
                    except:
                        feature_vector[col] = 0

            # Fill missing values
            feature_vector = feature_vector.fillna(0)

            # Get predictions
            predictions = {}
            probabilities = {}

            for model_name, model in self.models.items():
                try:
                    pred = model.predict(feature_vector)[0]
                    prob = model.predict_proba(feature_vector)[0]
                    predictions[model_name] = pred
                    probabilities[model_name] = {
                        'under_prob': prob[0],
                        'over_prob': prob[1]
                    }
                except Exception as e:
                    continue

            if not predictions:
                return None

            # Calculate ensemble with confidence weighting
            over_votes = sum(1 for pred in predictions.values() if pred == 1)
            total_votes = len(predictions)
            over_percentage = over_votes / total_votes

            confidence = features.get('prediction_confidence', 0.5)
            avg_over_prob = np.mean([prob['over_prob'] for prob in probabilities.values()])

            # Final recommendation with confidence adjustment
            recommendation = "OVER" if over_percentage > 0.5 else "UNDER"
            final_confidence = avg_over_prob if recommendation == "OVER" else 1 - avg_over_prob
            final_confidence = min(final_confidence * (0.5 + confidence * 0.5), 1.0)  # Adjust by prediction confidence

            # Extract insights
            scoring_pred = features.get('comprehensive_scoring_prediction', prop_line)
            line_diff = scoring_pred - prop_line

            insights = {
                'predicted_points': round(scoring_pred, 1),
                'line_vs_prediction': f"{line_diff:+.1f}",
                'confidence_level': self._get_confidence_level(final_confidence),
                'shot_quality': features.get('shot_shot_quality_score', 0.5),
                'clutch_factor': features.get('shot_clutch_closing_ability', 0.075),
                'spacing_bonus': features.get('shot_spacing_player_spacing_benefit', 0),
                'usage_role': features.get('shot_usage_usage_type', 'secondary_scorer'),
                'key_advantages': self._identify_v2_advantages(features, line_diff)
            }

            result = {
                'player_name': player_name,
                'market_type': market_type,
                'prop_line': prop_line,
                'recommendation': recommendation,
                'confidence': final_confidence,
                'over_percentage': over_percentage,
                'models_agreement': f"{over_votes}/{total_votes}",
                'v2_insights': insights
            }

            return result

        except Exception as e:
            print(f"❌ Error in v2 prediction: {e}")
            return None

    def _get_confidence_level(self, confidence):
        """Convert confidence to descriptive level."""
        if confidence >= 0.85:
            return "Very High"
        elif confidence >= 0.75:
            return "High"
        elif confidence >= 0.65:
            return "Medium"
        else:
            return "Low"

    def _identify_v2_advantages(self, features, line_diff):
        """Identify key advantages from v2 features."""
        advantages = []

        # Shot quality advantages
        if features.get('shot_shot_quality_score', 0) > 0.6:
            advantages.append("Excellent shot selection")

        if features.get('shot_recent_ts_trend', 0) > 0.01:
            advantages.append("Improving efficiency")

        # Clutch advantages
        if features.get('shot_clutch_clutch_trend', 0) > 2:
            advantages.append("Strong closer")

        # Spacing advantages
        if features.get('shot_spacing_team_shooter_count', 0) >= 4:
            advantages.append("Elite spacing")

        # Usage efficiency
        if features.get('shot_usage_usage_type') == 'high_efficiency_star':
            advantages.append("Star efficiency")

        # Line value
        if abs(line_diff) > 5:
            advantages.append("Significant line value")

        # Opponent matchup
        if features.get('def_opp_def_rating', 110) < 105:
            advantages.append("Favorable matchup")

        return advantages if advantages else ["Neutral factors"]

    def run_v2_predictions(self):
        """Run enhanced predictions with v1 + v2 features."""

        try:
            # Load models and data
            self.load_models_and_data()

            # Fetch today's player props
            print("\n=== FETCHING TODAY'S PLAYER PROPS ===")
            props_df = self.odds_client.get_all_todays_player_props()

            if props_df.empty:
                print("❌ No player props available")
                return None

            # Format props
            formatted_props = self.odds_client.format_for_ml_pipeline(props_df)
            print(f"✅ Formatted {len(formatted_props)} prop lines")

            # Generate predictions
            print(f"\n=== GENERATING V2 PREDICTIONS (ADVANCED SHOT QUALITY) ===")
            predictions = []

            for _, prop in formatted_props.iterrows():
                print(f"\nAnalyzing: {prop['fullName']} {prop['market_type']} O/U {prop['prop_line']}")

                opponent_team = prop['away_team'] if prop['home_team'] == prop.get('playerteamName', prop['home_team']) else prop['home_team']

                game_context = {
                    'game_date': prop['gameDate'],
                    'home_team': prop['home_team'],
                    'away_team': prop['away_team'],
                    'player_team': prop.get('playerteamName', prop['home_team']),
                    'opponent_team': opponent_team
                }

                prediction = self.make_v2_prediction(
                    prop['fullName'],
                    prop['prop_line'],
                    prop['market_type'],
                    game_context
                )

                if prediction:
                    prediction.update({
                        'over_odds': prop['over_odds'],
                        'bookmaker': prop['bookmaker'],
                        'game_time': prop['game_time']
                    })
                    predictions.append(prediction)

            if not predictions:
                print("❌ No predictions generated")
                return None

            # Create DataFrame and filter
            results_df = pd.DataFrame(predictions)
            high_confidence = results_df[results_df['confidence'] > 0.7].sort_values('confidence', ascending=False)

            # Display results
            self.display_v2_results(high_confidence)

            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            output_file = f"../data/processed/enhanced_v2_predictions_{timestamp}.csv"
            results_df.to_csv(output_file, index=False)
            print(f"\n✅ Saved v2 predictions to: {output_file}")

            return high_confidence

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def display_v2_results(self, predictions_df):
        """Display v2 predictions with detailed insights."""

        if predictions_df.empty:
            print("\n🤷 No high-confidence v2 predictions")
            return

        print(f"\n🎯 ADVANCED BETTING RECOMMENDATIONS - ITERATION 2 ({len(predictions_df)} total)")
        print("=" * 120)

        for _, pred in predictions_df.iterrows():
            insights = pred['v2_insights']

            print(f"\n🏀 {pred['player_name']} - {pred['market_type'].upper()}")
            print(f"   Line: {pred['prop_line']} | Predicted: {insights['predicted_points']}")
            print(f"   Line Value: {insights['line_vs_prediction']} points")
            print(f"   Game: vs {pred.get('opponent_team', 'TBD')}")
            print(f"   📊 RECOMMENDATION: {pred['recommendation']}")
            print(f"   📈 Confidence: {pred['confidence']:.1%} ({insights['confidence_level']})")
            print(f"   🤝 Model Agreement: {pred['models_agreement']}")
            print(f"   💰 Odds: {pred['over_odds']:+}")
            print(f"\n   🔍 V2 ANALYSIS:")
            print(f"      • Shot Quality: {insights['shot_quality']:.1%} (excellent > 60%)")
            print(f"      • Usage Role: {insights['usage_role'].replace('_', ' ').title()}")
            print(f"      • Clutch Factor: {insights['clutch_factor']:.3f}")
            print(f"      • Key Advantages: {', '.join(insights['key_advantages'])}")

        print("\n" + "=" * 120)

def main():
    """Run enhanced predictions v2."""

    if not os.getenv('ODDS_API_KEY'):
        print("❌ Missing API key!")
        return

    try:
        predictor = EnhancedPredictorV2()
        predictions = predictor.run_v2_predictions()

        if predictions is not None and not predictions.empty:
            print(f"\n🎉 V2 predictions complete! Found {len(predictions)} high-confidence recommendations.")
        else:
            print("\n📊 Analysis complete. No high-confidence predictions.")

    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()