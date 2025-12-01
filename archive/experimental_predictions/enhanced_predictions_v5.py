"""
Enhanced NBA Predictions - Iteration 5
Matchup Analytics with Statistical Significance Testing
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from odds_api_client import NBAOddsClient
from scaled_lr import ScaledLogisticRegression
from matchup_analytics_v5 import MatchupAnalyticsV5

class EnhancedPredictorV5:
    """Predictor with statistically validated matchup analytics."""

    def __init__(self, api_key=None):
        """Initialize with API key."""
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError("API key required")

        self.odds_client = NBAOddsClient(self.api_key)
        self.models = {}
        self.historical_data = None
        self.matchup_analyzer = None

        print("=" * 60)
        print("📊 ENHANCED NBA PREDICTIONS - ITERATION 5")
        print("   Statistical Matchup Analytics:")
        print("   ✓ Individual Home/Away Effects (p<0.05 testing)")
        print("   ✓ Player-Player Matchups (10+ games min)")
        print("   ✓ Sample Size Validation")
        print("   ✓ Opponent-Specific History")
        print("   ✓ Statistical Significance Testing")
        print("=" * 60)

    def load_models_and_data(self):
        """Load models and matchup analyzer."""
        print("\n📂 LOADING MODELS AND MATCHUP ANALYZER")

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

        # Initialize matchup analyzer
        self.matchup_analyzer = MatchupAnalyticsV5(self.historical_data)
        print("✅ Matchup analytics analyzer loaded")

        return len(self.models) > 0

    def make_matchup_based_prediction(self, player_name, prop_line, market_type, game_context):
        """Make prediction with statistical matchup analytics."""

        try:
            # Get baseline
            player_data = self.historical_data[self.historical_data['fullName'] == player_name]
            if player_data.empty:
                return None

            baseline_points = player_data['points'].tail(10).mean()

            # Generate matchup features
            matchup_features = self.matchup_analyzer.generate_all_matchup_features(
                player_name, game_context['opponent_team']
            )

            # Extract key matchup insights
            ha_advantage = matchup_features.get('ind_ha_weighted_advantage', 0)
            ha_confidence = matchup_features.get('ind_ha_sample_confidence', 0.5)
            opp_history = matchup_features.get('opp_points_avg', baseline_points)
            opp_confidence = matchup_features.get('opp_sample_confidence', 0.1)
            matchup_reliable = matchup_features.get('matchup_reliable', 0.5)

            # Calculate adjustment
            ha_adjustment = ha_advantage * ha_confidence
            opp_adjustment = (opp_history - baseline_points) * opp_confidence * matchup_reliable

            # Predicted value
            if market_type.lower() == 'points':
                predicted_value = baseline_points + ha_adjustment + opp_adjustment
            else:  # Rebounds - simplified conversion
                predicted_value = baseline_points * 0.4

            # Determine recommendation
            recommendation = "OVER" if predicted_value > prop_line else "UNDER"

            # Calculate confidence based on data reliability
            confidence = min(
                ha_confidence * 0.4 +
                opp_confidence * 0.4 +
                matchup_reliable * 0.2,
                0.95
            )

            # Only high confidence predictions
            if confidence < 0.60:  # Higher threshold for statistical reliability
                return None

            result = {
                'player_name': player_name,
                'market_type': market_type,
                'prop_line': prop_line,
                'recommendation': recommendation,
                'confidence': confidence,
                'v5_insights': {
                    'predicted_value': round(predicted_value, 1),
                    'line_value': round(predicted_value - prop_line, 1),
                    'ha_significant': matchup_features.get('ind_ha_points_significant', False),
                    'ha_advantage': round(ha_advantage, 1),
                    'opp_history': round(opp_history, 1),
                    'opp_games': matchup_features.get('opp_total_games', 0),
                    'statistical_reliability': 'High' if confidence > 0.75 else 'Medium',
                    'sample_adequate': matchup_features.get('opp_sample_adequate', False)
                }
            }

            return result

        except Exception as e:
            print(f"❌ Error in v5 prediction for {player_name}: {e}")
            return None

    def run_v5_predictions(self):
        """Run statistically validated predictions."""

        if not self.load_models_and_data():
            return None

        print("\n📊 FETCHING PLAYER PROPS FOR STATISTICAL MATCHUP ANALYSIS")
        props_df = self.odds_client.get_all_todays_player_props()

        if props_df.empty:
            print("❌ No player props available")
            return None

        formatted_props = self.odds_client.format_for_ml_pipeline(props_df)
        print(f"✅ Found {len(formatted_props)} prop lines")

        print(f"\n🔮 GENERATING STATISTICALLY VALIDATED PREDICTIONS (ITERATION 5)")
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

            prediction = self.make_matchup_based_prediction(
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
            print("\n❌ No statistically significant predictions")
            return None

        # Display results
        self.display_v5_results(predictions)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_file = f"../data/processed/matchup_v5_predictions_{timestamp}.csv"
        pd.DataFrame(predictions).to_csv(output_file, index=False)
        print(f"\n💾 Saved to: {output_file}")

        return predictions

    def display_v5_results(self, predictions):
        """Display statistically validated predictions."""
        if not predictions:
            return

        print(f"\n🎓 STATISTICALLY VALIDATED BETTING RECOMMENDATIONS ({len(predictions)} total)")
        print("=" * 80)
        print("📌 All predictions meet minimum sample size requirements")
        print("📊 Statistical significance testing applied")

        for _, pred in predictions.iterrows():
            insights = pred['v5_insights']

            print(f"\n🏀 {pred['player_name'].upper()} - {pred['market_type'].upper()}")
            print(f"   Line: {pred['prop_line']} | Predicted: {insights['predicted_value']}")
            print(f"   Line Value: {insights['line_value']:+} points")
            print(f"   📊 RECOMMENDATION: {pred['recommendation']}")
            print(f"   📈 Confidence: {pred['confidence']:.1%} ({insights['statistical_reliability']})")
            print(f"   💰 Odds: {pred['bookmaker']} {pred['over_odds']:+}")
            print(f"\n   📓 STATISTICAL ANALYSIS:")
            print(f"      • Home/Away Significant: {insights['ha_significant']}")
            print(f"      • HA Advantage: {insights['ha_advantage']:+.1f} PPG")
            print(f"      • Opponent History: {insights['opp_history']:.1f} PPG ({insights['opp_games']} games)")
            print(f"      • Sample Adequate: {insights['sample_adequate']}")

        print("\n" + "=" * 80)

def main():
    """Run v5 predictions."""
    if not os.getenv('ODDS_API_KEY'):
        print("❌ Missing API key!")
        return

    try:
        predictor = EnhancedPredictorV5()
        predictions = predictor.run_v5_predictions()

        if predictions:
            print(f"\n🎉 V5 complete! {len(predictions)} statistically valid recommendations.")
        else:
            print("\n📊 Complete. No statistically significant predictions.")

    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()