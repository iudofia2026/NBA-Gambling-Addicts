"""
Daily NBA Player Props Predictions

This script automatically:
1. Fetches today's NBA player prop lines using The Odds API
2. Loads our trained ML models
3. Generates features for each player's upcoming game
4. Makes over/under predictions with confidence scores
5. Outputs recommended bets for the day

Usage:
    python src/daily_predictions.py

Environment Variables:
    ODDS_API_KEY: Your The Odds API key (required)
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import warnings
import joblib
warnings.filterwarnings('ignore')

# Import our modules
from odds_api_client import NBAOddsClient
from scaled_lr import ScaledLogisticRegression

class DailyPredictor:
    """Main class for generating daily NBA prop predictions."""

    def __init__(self, api_key=None):
        """Initialize the daily predictor."""
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError("API key required. Set ODDS_API_KEY environment variable.")

        self.odds_client = NBAOddsClient(self.api_key)
        self.models = {}
        self.feature_cols = None
        self.label_encoders = {}

        print("=== NBA DAILY PREDICTIONS SYSTEM ===")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def load_models(self):
        """Load all trained ML models and preprocessing components."""
        print("\n=== LOADING TRAINED MODELS ===")

        model_dir = '../models'

        try:
            # Load models
            model_files = {
                'logistic_regression': 'logistic_regression_model.pkl',
                'random_forest': 'random_forest_model.pkl',
                'xgboost': 'xgboost_model.pkl'
            }

            for model_name, filename in model_files.items():
                filepath = f"{model_dir}/{filename}"
                if os.path.exists(filepath):
                    self.models[model_name] = joblib.load(filepath)
                    print(f"✓ Loaded {model_name} model")
                else:
                    print(f"⚠️  {model_name} model not found: {filepath}")

            # Load feature columns
            feature_file = f"{model_dir}/feature_columns.pkl"
            if os.path.exists(feature_file):
                self.feature_cols = joblib.load(feature_file)
                print(f"✓ Loaded {len(self.feature_cols)} feature columns")
            else:
                raise FileNotFoundError(f"Feature columns not found: {feature_file}")

            # Load label encoders
            encoders_file = f"{model_dir}/label_encoders.pkl"
            if os.path.exists(encoders_file):
                self.label_encoders = joblib.load(encoders_file)
                print(f"✓ Loaded {len(self.label_encoders)} label encoders")
            else:
                print("⚠️  No label encoders found (may not be needed)")

            if not self.models:
                raise ValueError("No models loaded successfully!")

            print(f"\n✅ Successfully loaded {len(self.models)} models")

        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise

    def get_historical_player_data(self):
        """Load historical player data for feature generation."""
        print("\n=== LOADING HISTORICAL PLAYER DATA ===")

        try:
            # Load our full historical dataset
            data = pd.read_csv('../data/processed/engineered_features.csv')
            data['gameDate'] = pd.to_datetime(data['gameDate'], errors='coerce')

            print(f"✓ Loaded {len(data):,} historical game records")
            print(f"✓ Date range: {data['gameDate'].min().date()} to {data['gameDate'].max().date()}")
            print(f"✓ Unique players: {data['fullName'].nunique()}")

            return data

        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
            raise

    def generate_game_context_features(self, player_name, game_date, home_team, away_team, historical_data):
        """Generate features for a player's upcoming game."""

        # Filter player's historical data
        player_data = historical_data[historical_data['fullName'] == player_name].copy()

        if player_data.empty:
            print(f"⚠️  No historical data for {player_name}")
            return None

        # Sort by date to get most recent performance
        player_data = player_data.sort_values('gameDate')

        # Get the most recent game features as baseline
        latest_game = player_data.iloc[-1].copy()

        # Update game-specific context
        latest_game['gameDate'] = pd.to_datetime(game_date)
        latest_game['year'] = pd.to_datetime(game_date).year

        # Update team context (simplified - in production you'd want more sophisticated opponent analysis)
        # For now, we'll use the player's recent averages

        return latest_game

    def make_prediction_for_prop(self, player_name, prop_line, market_type, game_context, historical_data):
        """Make a prediction for a specific player prop."""

        try:
            # Generate features for this player's upcoming game
            features = self.generate_game_context_features(
                player_name, game_context['game_date'],
                game_context['home_team'], game_context['away_team'],
                historical_data
            )

            if features is None:
                return None

            # Prepare feature vector for prediction
            feature_vector = features[self.feature_cols].to_frame().T

            # Handle categorical features
            for col, encoder in self.label_encoders.items():
                if col in feature_vector.columns:
                    try:
                        feature_vector[col] = encoder.transform(feature_vector[col].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        feature_vector[col] = 0

            # Fill missing values
            feature_vector = feature_vector.fillna(0)

            # Get predictions from all models
            predictions = {}
            probabilities = {}

            for model_name, model in self.models.items():
                try:
                    # Get prediction and probability
                    pred = model.predict(feature_vector)[0]
                    prob = model.predict_proba(feature_vector)[0]

                    predictions[model_name] = pred
                    probabilities[model_name] = {
                        'under_prob': prob[0],
                        'over_prob': prob[1]
                    }

                except Exception as e:
                    print(f"⚠️  Error with {model_name} prediction for {player_name}: {e}")
                    continue

            if not predictions:
                return None

            # Create ensemble prediction (majority vote with confidence)
            over_votes = sum(1 for pred in predictions.values() if pred == 1)
            total_votes = len(predictions)
            over_percentage = over_votes / total_votes

            # Calculate average confidence
            avg_over_prob = np.mean([prob['over_prob'] for prob in probabilities.values()])
            avg_under_prob = np.mean([prob['under_prob'] for prob in probabilities.values()])

            # Final recommendation
            if over_percentage > 0.5:
                recommendation = "OVER"
                confidence = avg_over_prob
            else:
                recommendation = "UNDER"
                confidence = avg_under_prob

            return {
                'player_name': player_name,
                'market_type': market_type,
                'prop_line': prop_line,
                'recommendation': recommendation,
                'confidence': confidence,
                'over_percentage': over_percentage,
                'models_agreement': f"{over_votes}/{total_votes}",
                'individual_predictions': predictions,
                'individual_probabilities': probabilities
            }

        except Exception as e:
            print(f"❌ Error making prediction for {player_name}: {e}")
            return None

    def run_daily_predictions(self):
        """Main method to run daily predictions."""

        try:
            # Step 1: Load models
            self.load_models()

            # Step 2: Load historical data
            historical_data = self.get_historical_player_data()

            # Step 3: Fetch today's player props
            print("\n=== FETCHING TODAY'S PLAYER PROPS ===")
            props_df = self.odds_client.get_all_todays_player_props()

            if props_df.empty:
                print("❌ No player props available for today")
                return None

            # Step 4: Format props data
            formatted_props = self.odds_client.format_for_ml_pipeline(props_df)

            # Step 5: Generate predictions for each prop
            print(f"\n=== GENERATING PREDICTIONS FOR {len(formatted_props)} PROPS ===")

            predictions = []

            for _, prop in formatted_props.iterrows():
                print(f"\nAnalyzing: {prop['fullName']} {prop['market_type']} O/U {prop['prop_line']}")

                game_context = {
                    'game_date': prop['gameDate'],
                    'home_team': prop['home_team'],
                    'away_team': prop['away_team']
                }

                prediction = self.make_prediction_for_prop(
                    prop['fullName'],
                    prop['prop_line'],
                    prop['market_type'],
                    game_context,
                    historical_data
                )

                if prediction:
                    # Add odds and bookmaker info
                    prediction.update({
                        'over_odds': prop['over_odds'],
                        'bookmaker': prop['bookmaker'],
                        'home_team': prop['home_team'],
                        'away_team': prop['away_team'],
                        'game_time': prop['game_time']
                    })
                    predictions.append(prediction)

            if not predictions:
                print("❌ No predictions generated")
                return None

            # Step 6: Create results DataFrame and analysis
            results_df = pd.DataFrame(predictions)

            # Step 7: Filter for high-confidence predictions
            print(f"\n=== FILTERING HIGH-CONFIDENCE PREDICTIONS ===")

            # Only show predictions with >60% confidence and good model agreement
            high_confidence = results_df[
                (results_df['confidence'] > 0.6) &
                (results_df['over_percentage'].isin([0.0, 1.0]))  # All models agree
            ].copy()

            # Sort by confidence
            high_confidence = high_confidence.sort_values('confidence', ascending=False)

            # Step 8: Display results
            self.display_predictions(high_confidence)

            # Step 9: Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            output_file = f"../data/processed/daily_predictions_{timestamp}.csv"
            results_df.to_csv(output_file, index=False)

            print(f"\n✅ Saved full predictions to: {output_file}")
            print(f"✅ Generated {len(results_df)} total predictions")
            print(f"✅ {len(high_confidence)} high-confidence recommendations")

            return high_confidence

        except Exception as e:
            print(f"❌ Error in daily predictions: {e}")
            raise

    def display_predictions(self, predictions_df):
        """Display predictions in a user-friendly format."""

        if predictions_df.empty:
            print("\n🤷 No high-confidence predictions for today")
            return

        print(f"\n🎯 HIGH-CONFIDENCE BETTING RECOMMENDATIONS ({len(predictions_df)} total)")
        print("=" * 80)

        for _, pred in predictions_df.iterrows():
            print(f"\n🏀 {pred['player_name']} - {pred['market_type'].upper()}")
            print(f"   Line: {pred['prop_line']}")
            print(f"   Game: {pred['away_team']} @ {pred['home_team']}")
            print(f"   📊 RECOMMENDATION: {pred['recommendation']}")
            print(f"   📈 Confidence: {pred['confidence']:.1%}")
            print(f"   🤝 Model Agreement: {pred['models_agreement']}")
            print(f"   💰 Odds: {pred['over_odds']:+}")
            print(f"   🏪 Book: {pred['bookmaker']}")

        print("\n" + "=" * 80)
        print("⚠️  DISCLAIMER: These are model predictions for educational purposes only.")
        print("    Always do your own research and gamble responsibly.")

def main():
    """Main function to run daily predictions."""

    # Check for API key
    if not os.getenv('ODDS_API_KEY'):
        print("❌ Missing API key!")
        print("\nTo use this system:")
        print("1. Sign up at https://the-odds-api.com/")
        print("2. Get your free API key (500 requests/month)")
        print("3. Set environment variable:")
        print("   export ODDS_API_KEY='your_api_key_here'")
        print("\nThen run: python src/daily_predictions.py")
        return

    try:
        # Initialize and run predictor
        predictor = DailyPredictor()
        predictions = predictor.run_daily_predictions()

        if predictions is not None and not predictions.empty:
            print(f"\n🎉 Daily predictions complete! Found {len(predictions)} high-confidence bets.")
        else:
            print("\n📊 Daily analysis complete. No high-confidence predictions today.")

    except Exception as e:
        print(f"\n💥 Error running daily predictions: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()