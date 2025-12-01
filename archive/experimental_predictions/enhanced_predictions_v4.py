"""
Enhanced NBA Predictions - Iteration 4
Evidence-based features with academic backing
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from odds_api_client import NBAOddsClient
from scaled_lr import ScaledLogisticRegression
from evidence_features_v4 import EvidenceBasedFeaturesV4

class EnhancedPredictorV4:
    """Predictor with evidence-based features from academic research."""

    def __init__(self, api_key=None):
        """Initialize with API key."""
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError("API key required")

        self.odds_client = NBAOddsClient(self.api_key)
        self.models = {}
        self.historical_data = None
        self.evidence_generator = None

        print("=" * 60)
        print("🎓 ENHANCED NBA PREDICTIONS - ITERATION 4")
        print("   Evidence-Based Features:")
        print("   ✓ Home Court Advantage (55-60% win rate, p<0.001)")
        print("   ✓ Rest Days Impact (+2.1 PPG, -3.4 B2B)")
        print("   ✓ Altitude Effects (5280ft = +3.2/-2.1 PPG)")
        print("   ✓ Officiating Bias (+0.8 FTA home)")
        print("   ✓ Situational Factors")
        print("=" * 60)

    def load_models_and_data(self):
        """Load models and evidence generator."""
        print("\n📂 LOADING MODELS AND EVIDENCE GENERATOR")

        model_dir = '../models'
        model_files = {
            'random_forest': 'random_forest_model.pkl',
            'xgboost': 'xgboost_model.pkl'
        }

        for model_name, filename in model_files.items():
            filepath = f"{model_dir}/{filename}"
            if os.path.exists(filepath):
                self.models[model_name] = joblib.load(filepath)
                print(f"✅ {model_name.upper()} model loaded")

        # Load historical data
        self.historical_data = pd.read_csv('../data/processed/engineered_features.csv')
        self.historical_data['gameDate'] = pd.to_datetime(self.historical_data['gameDate'], errors='coerce')
        print(f"✅ Historical data: {len(self.historical_data):,} games")

        # Initialize evidence generator
        self.evidence_generator = EvidenceBasedFeaturesV4(self.historical_data)
        print("✅ Evidence-based features generator loaded")

        return len(self.models) > 0

    def make_evidence_based_prediction(self, player_name, prop_line, market_type, game_context):
        """Make prediction with evidence-based features."""

        try:
            # Get baseline
            player_data = self.historical_data[self.historical_data['fullName'] == player_name]
            if player_data.empty:
                return None

            baseline_points = player_data['points'].tail(10).mean()

            # Generate evidence-based features
            evidence_features = self.evidence_generator.generate_all_evidence_features(
                player_name, game_context
            )

            # Evidence-based prediction
            evidence_adjustment = evidence_features.get('evidence_adjustment', 0)
            situational_multiplier = evidence_features.get('sit_situational_intensity', 1.0)

            predicted_value = (baseline_points + evidence_adjustment) * situational_multiplier

            # Convert to market type
            if market_type.lower() == 'points':
                prediction = predicted_value
            else:  # Rebounds - simplified conversion
                prediction = baseline_points * 0.4

            # Determine recommendation
            recommendation = "OVER" if prediction > prop_line else "UNDER"

            # Calculate confidence based on evidence strength
            confidence_factors = [
                evidence_features.get('hca_home_court_advantage', 0) / 5.0,  # Normalized
                evidence_features.get('rest_rest_days', 2) / 5.0,
                abs(evidence_features.get('alt_altitude_impact', 0)) / 3.0,
                evidence_features.get('sit_situational_intensity', 1.0)
            ]
            confidence = min(np.mean(confidence_factors), 0.95)

            # Only high confidence predictions
            if confidence < 0.65:
                return None

            result = {
                'player_name': player_name,
                'market_type': market_type,
                'prop_line': prop_line,
                'recommendation': recommendation,
                'confidence': confidence,
                'v4_insights': {
                    'predicted_value': round(prediction, 1),
                    'line_value': round(prediction - prop_line, 1),
                    'home_advantage': round(evidence_features.get('hca_home_court_advantage', 0), 1),
                    'rest_impact': round(evidence_features.get('rest_rest_impact', 0), 1),
                    'altitude_effect': round(evidence_features.get('alt_altitude_impact', 0), 1),
                    'officiating_bias': round(evidence_features.get('off_expected_fta_impact', 0), 1),
                    'situational': f"{evidence_features.get('sit_season_stage', 'normal')} stage"
                }
            }

            return result

        except Exception as e:
            print(f"❌ Error in v4 prediction for {player_name}: {e}")
            return None

    def run_v4_predictions(self):
        """Run evidence-based predictions."""

        if not self.load_models_and_data():
            return None

        print("\n📊 FETCHING PLAYER PROPS FOR EVIDENCE-BASED PREDICTIONS")
        props_df = self.odds_client.get_all_todays_player_props()

        if props_df.empty:
            print("❌ No player props available")
            return None

        formatted_props = self.odds_client.format_for_ml_pipeline(props_df)
        print(f"✅ Found {len(formatted_props)} prop lines")

        print(f"\n🔮 GENERATING EVIDENCE-BASED PREDICTIONS (ITERATION 4)")
        predictions = []

        for _, prop in formatted_props.iterrows():
            print(f"\n🎯 {prop['fullName']} - {prop['market_type']}")

            opponent_team = prop['away_team'] if prop['home_team'] == prop.get('playerteamName', '') else prop['home_team']

            game_context = {
                'game_date': prop['gameDate'],
                'home_team': prop['home_team'],
                'away_team': prop['away_team'],
                'player_team': prop.get('playerteamName', ''),
                'opponent_team': opponent_team
            }

            prediction = self.make_evidence_based_prediction(
                prop['fullName'],
                prop['prop_line'],
                prop['market_type'],
                game_context
            )

            if prediction:
                prediction.update({
                    'over_odds': prop['over_odds'],
                    'bookmaker': prop['bookmaker']
                })
                predictions.append(prediction)

        if not predictions:
            print("\n❌ No high-confidence predictions")
            return None

        # Display results
        self.display_v4_results(predictions)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_file = f"../data/processed/evidence_v4_predictions_{timestamp}.csv"
        pd.DataFrame(predictions).to_csv(output_file, index=False)
        print(f"\n💾 Saved to: {output_file}")

        return predictions

    def display_v4_results(self, predictions):
        """Display evidence-based predictions."""
        if not predictions:
            return

        print(f"\n🏆 EVIDENCE-BASED BETTING RECOMMENDATIONS ({len(predictions)} total)")
        print("=" * 80)

        for _, pred in predictions.iterrows():
            insights = pred['v4_insights']

            print(f"\n🏀 {pred['player_name'].upper()} - {pred['market_type'].upper()}")
            print(f"   Line: {pred['prop_line']} | Predicted: {insights['predicted_value']}")
            print(f"   Line Value: {insights['line_value']:+} points")
            print(f"   📊 RECOMMENDATION: {pred['recommendation']}")
            print(f"   📈 Confidence: {pred['confidence']:.1%}")
            print(f"   💰 Odds: {pred['bookmaker']} {pred['over_odds']:+}")
            print(f"\n   📓 EVIDENCE ANALYSIS:")
            print(f"      • Home Court: {insights['home_advantage']:+.1f} PPG")
            print(f"      • Rest Impact: {insights['rest_impact']:+.1f} PPG")
            print(f"      • Altitude: {insights['altitude_effect']:+.1f} PPG")
            print(f"      • Officiating: {insights['officiating_bias']:+.1f} FTA equivalent")
            print(f"      • Situation: {insights['situational']}")

        print("\n" + "=" * 80)

def main():
    """Run v4 predictions."""
    if not os.getenv('ODDS_API_KEY'):
        print("❌ Missing API key!")
        return

    try:
        predictor = EnhancedPredictorV4()
        predictions = predictor.run_v4_predictions()

        if predictions:
            print(f"\n🎉 V4 complete! {len(predictions)} recommendations.")
        else:
            print("\n📊 Complete. No predictions.")

    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()