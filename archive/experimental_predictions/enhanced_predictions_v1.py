"""
Enhanced NBA Predictions - Iteration 1
Integration of weighted performance windows, defensive matchups, and external factors
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

class EnhancedPredictorV1:
    """Enhanced predictor with Iteration 1 features."""

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

    def load_models_and_data(self):
        """Load models and initialize feature generators."""
        print("\n=== LOADING MODELS AND DATA ===")

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

        # Initialize feature generators
        self.advanced_generator = AdvancedFeatureGeneratorV1(self.historical_data)
        self.external_generator = ExternalFactorsV1(self.historical_data)

        print("✅ Feature generators initialized")

    def prepare_enhanced_features(self, player_name, game_context):
        """Prepare all enhanced features for prediction."""

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

        # Generate advanced features
        advanced_features = self.advanced_generator.generate_all_advanced_features(
            player_name, game_context['opponent_team']
        )
        for feature, value in advanced_features.items():
            if feature in self.feature_cols:
                baseline[feature] = value
            else:
                self.feature_cols = list(self.feature_cols) + [feature]
                baseline[feature] = value

        # Generate external factors
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

        # Create composite feature
        baseline['enhanced_scoring_prediction'] = (
            advanced_features.get('scoring_expectation', baseline.get('points', 0)) +
            external_features.get('external_scoring_adjustment', 0)
        )

        return baseline

    def make_enhanced_prediction(self, player_name, prop_line, market_type, game_context):
        """Make prediction with all enhanced features."""

        try:
            # Prepare features
            features = self.prepare_enhanced_features(player_name, game_context)
            if features is None:
                return None

            # Prepare feature vector
            feature_vector = features[self.feature_cols].to_frame().T

            # Handle categorical features
            for col, encoder in self.label_encoders.items():
                if col in feature_vector.columns:
                    try:
                        # Handle new categories
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
                    print(f"⚠️  {model_name} prediction failed: {e}")
                    continue

            if not predictions:
                return None

            # Calculate ensemble
            over_votes = sum(1 for pred in predictions.values() if pred == 1)
            total_votes = len(predictions)
            over_percentage = over_votes / total_votes

            avg_over_prob = np.mean([prob['over_prob'] for prob in probabilities.values()])

            # Final recommendation
            recommendation = "OVER" if over_percentage > 0.5 else "UNDER"
            confidence = avg_over_prob if recommendation == "OVER" else 1 - avg_over_prob

            # Get insights
            scoring_pred = features.get('enhanced_scoring_prediction', prop_line)
            matchup_quality = features.get('matchup_quality_score', 0)
            travel_impact = features.get('travel_external_scoring_adjustment', 0)

            result = {
                'player_name': player_name,
                'market_type': market_type,
                'prop_line': prop_line,
                'recommendation': recommendation,
                'confidence': confidence,
                'over_percentage': over_percentage,
                'models_agreement': f"{over_votes}/{total_votes}",
                'enhanced_insights': {
                    'predicted_points': round(scoring_pred, 1),
                    'line_vs_prediction': round(scoring_pred - prop_line, 1),
                    'matchup_quality': round(matchup_quality, 2),
                    'travel_impact': round(travel_impact, 1),
                    'key_factors': self._identify_key_factors(features)
                }
            }

            return result

        except Exception as e:
            print(f"❌ Error in enhanced prediction: {e}")
            return None

    def _identify_key_factors(self, features):
        """Identify the most influential factors for this prediction."""
        factors = []

        # Performance trends
        if features.get('weighted_points_trend', 0) > 2:
            factors.append("Hot scoring streak")
        elif features.get('weighted_points_trend', 0) < -2:
            factors.append("Cold scoring streak")

        # Defensive matchup
        if features.get('def_opp_def_rating', 110) < 105:
            factors.append("Weak opponent defense")
        elif features.get('def_opp_def_rating', 110) > 115:
            factors.append("Strong opponent defense")

        # Travel fatigue
        if features.get('travel_is_back_to_back', False):
            factors.append("Back-to-back game")
        elif features.get('travel_travel_hours', 0) > 4:
            factors.append("Long travel")

        # Injury risk
        if features.get('injury_injury_risk', 0) > 0.2:
            factors.append("Potential injury concern")

        # Pace factor
        if features.get('team_pace_factor', 1) > 1.05:
            factors.append("Fast-paced game expected")
        elif features.get('team_pace_factor', 1) < 0.95:
            factors.append("Slow-paced game expected")

        return factors if factors else ["Normal conditions"]

    def run_enhanced_predictions(self):
        """Run enhanced predictions with all Iteration 1 features."""

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
            print(f"\n=== GENERATING ENHANCED PREDICTIONS (ITERATION 1) ===")
            predictions = []

            for _, prop in formatted_props.iterrows():
                print(f"\nAnalyzing: {prop['fullName']} {prop['market_type']} O/U {prop['prop_line']}")

                # Determine teams (simplified)
                opponent_team = prop['away_team'] if prop['home_team'] == prop.get('playerteamName', prop['home_team']) else prop['home_team']

                game_context = {
                    'game_date': prop['gameDate'],
                    'home_team': prop['home_team'],
                    'away_team': prop['away_team'],
                    'player_team': prop.get('playerteamName', prop['home_team']),
                    'opponent_team': opponent_team
                }

                prediction = self.make_enhanced_prediction(
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
            high_confidence = results_df[
                (results_df['confidence'] > 0.65) &
                (results_df['over_percentage'].isin([0.0, 1.0]))
            ].sort_values('confidence', ascending=False)

            # Display results
            self.display_enhanced_results(high_confidence)

            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            output_file = f"../data/processed/enhanced_v1_predictions_{timestamp}.csv"
            results_df.to_csv(output_file, index=False)
            print(f"\n✅ Saved enhanced predictions to: {output_file}")

            return high_confidence

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def display_enhanced_results(self, predictions_df):
        """Display enhanced predictions with insights."""

        if predictions_df.empty:
            print("\n🤷 No high-confidence predictions")
            return

        print(f"\n🎯 ENHANCED BETTING RECOMMENDATIONS - ITERATION 1 ({len(predictions_df)} total)")
        print("=" * 100)

        for _, pred in predictions_df.iterrows():
            insights = pred['enhanced_insights']

            print(f"\n🏀 {pred['player_name']} - {pred['market_type'].upper()}")
            print(f"   Line: {pred['prop_line']} | Predicted: {insights['predicted_points']}")
            print(f"   Game: vs {pred.get('opponent_team', 'TBD')}")
            print(f"   📊 RECOMMENDATION: {pred['recommendation']}")
            print(f"   📈 Confidence: {pred['confidence']:.1%}")
            print(f"   🤝 Model Agreement: {pred['models_agreement']}")
            print(f"   💰 Odds: {pred['over_odds']:+} | Value: {insights['line_vs_prediction']:+.1f}")
            print(f"   🔍 Key Factors: {', '.join(insights['key_factors'])}")

        print("\n" + "=" * 100)

def main():
    """Run enhanced predictions v1."""

    if not os.getenv('ODDS_API_KEY'):
        print("❌ Missing API key!")
        print("\nSet ODDS_API_KEY environment variable")
        return

    try:
        predictor = EnhancedPredictorV1()
        predictions = predictor.run_enhanced_predictions()

        if predictions is not None and not predictions.empty:
            print(f"\n🎉 Enhanced predictions complete! Found {len(predictions)} recommendations.")
        else:
            print("\n📊 Analysis complete. No high-confidence predictions.")

    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()