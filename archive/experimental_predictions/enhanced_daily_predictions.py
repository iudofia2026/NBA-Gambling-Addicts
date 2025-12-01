"""
Enhanced NBA Player Props Predictions with Matchup Features

This script automatically:
1. Fetches today's NBA player prop lines using The Odds API
2. Loads our trained ML models
3. Generates matchup-specific features (player vs team history, opponent defense)
4. Makes over/under predictions with confidence scores considering matchups
5. Outputs recommended bets with matchup analysis
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

class EnhancedDailyPredictor:
    """Enhanced predictor with matchup-specific analysis."""

    def __init__(self, api_key=None):
        """Initialize the enhanced daily predictor."""
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError("API key required. Set ODDS_API_KEY environment variable.")

        self.odds_client = NBAOddsClient(self.api_key)
        self.models = {}
        self.feature_cols = None
        self.label_encoders = {}
        self.historical_data = None

        print("=== ENHANCED NBA DAILY PREDICTIONS SYSTEM ===")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("✅ Matchup-specific features enabled")

    def load_models_and_data(self):
        """Load models and historical data."""
        print("\n=== LOADING MODELS AND DATA ===")

        # Load models
        model_dir = '../models'
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

        # Load feature columns
        feature_file = f"{model_dir}/feature_columns.pkl"
        if os.path.exists(feature_file):
            self.feature_cols = joblib.load(feature_file)
            print(f"✓ Loaded {len(self.feature_cols)} feature columns")

        # Load label encoders
        encoders_file = f"{model_dir}/label_encoders.pkl"
        if os.path.exists(encoders_file):
            self.label_encoders = joblib.load(encoders_file)
            print(f"✓ Loaded {len(self.label_encoders)} label encoders")

        # Load historical data
        self.historical_data = pd.read_csv('../data/processed/engineered_features.csv')
        self.historical_data['gameDate'] = pd.to_datetime(self.historical_data['gameDate'], errors='coerce')

        print(f"✓ Loaded {len(self.historical_data):,} historical game records")

        if not self.models:
            raise ValueError("No models loaded successfully!")

        return True

    def get_player_vs_opponent_history(self, player_name, opponent_team, games_back=5):
        """Get player's historical performance against specific opponent."""

        # Filter games where player faced this opponent
        matchup_games = self.historical_data[
            (self.historical_data['fullName'] == player_name) &
            ((self.historical_data['playerteamName'] == opponent_team) |
             (self.historical_data['opponentteamName'] == opponent_team))
        ].sort_values('gameDate', ascending=False).head(games_back)

        if matchup_games.empty:
            return {
                'avg_points_vs_opponent': 0,
                'over_rate_vs_opponent': 0.5,
                'games_vs_opponent': 0
            }

        return {
            'avg_points_vs_opponent': matchup_games['points'].mean(),
            'over_rate_vs_opponent': (matchup_games['over_threshold'] == 1).mean(),
            'games_vs_opponent': len(matchup_games)
        }

    def get_opponent_defense_stats(self, opponent_team):
        """Get opponent's defensive statistics from historical data."""

        opponent_games = self.historical_data[self.historical_data['opponentteamName'] == opponent_team]

        if opponent_games.empty:
            return {
                'opponent_avg_points_allowed': 110,  # League average
                'opponent_def_strength': 0  # Neutral
            }

        avg_points_allowed = opponent_games['points'].mean()

        # Calculate defensive strength (negative = good defense)
        def_strength = (110 - avg_points_allowed) / 110

        return {
            'opponent_avg_points_allowed': avg_points_allowed,
            'opponent_def_strength': def_strength
        }

    def generate_matchup_features(self, player_name, opponent_team):
        """Generate matchup-specific features for prediction."""

        features = {}

        # Player vs opponent history
        matchup_history = self.get_player_vs_opponent_history(player_name, opponent_team)
        features.update({f'matchup_{k}': v for k, v in matchup_history.items()})

        # Opponent defense
        opponent_defense = self.get_opponent_defense_stats(opponent_team)
        features.update({f'opp_def_{k}': v for k, v in opponent_defense.items()})

        # Combined matchup advantage
        features['matchup_advantage'] = (
            matchup_history['avg_points_vs_opponent'] -
            opponent_defense['opponent_avg_points_allowed']
        )

        # Experience factor (more games = more reliable data)
        features['matchup_experience'] = min(matchup_history['games_vs_opponent'] / 10, 1.0)

        return features

    def make_enhanced_prediction(self, player_name, prop_line, market_type, game_context):
        """Make prediction with matchup analysis."""

        try:
            # Get historical baseline
            player_data = self.historical_data[self.historical_data['fullName'] == player_name]

            if player_data.empty:
                return None

            # Get most recent game features
            latest_game = player_data.sort_values('gameDate').iloc[-1].copy()
            latest_game['gameDate'] = pd.to_datetime(game_context['game_date'])
            latest_game['year'] = pd.to_datetime(game_context['game_date']).year

            # Update team context for this game
            latest_game['playerteamName'] = game_context.get('player_team', latest_game['playerteamName'])
            latest_game['opponentteamName'] = game_context['opponent_team']
            latest_game['home'] = (latest_game['playerteamName'] == game_context['home_team'])

            # Generate matchup features
            opponent_team = game_context['opponent_team']
            matchup_features = self.generate_matchup_features(player_name, opponent_team)

            # Add matchup features to the feature set
            for feature, value in matchup_features.items():
                if feature in self.feature_cols:
                    latest_game[feature] = value
                else:
                    # Add missing matchup features
                    self.feature_cols = list(self.feature_cols) + [feature]
                    latest_game[feature] = value

            # Prepare feature vector
            feature_vector = latest_game[self.feature_cols].to_frame().T

            # Handle categorical features
            for col, encoder in self.label_encoders.items():
                if col in feature_vector.columns:
                    try:
                        feature_vector[col] = encoder.transform(feature_vector[col].astype(str))
                    except ValueError:
                        feature_vector[col] = 0

            # Fill missing values
            feature_vector = feature_vector.fillna(0)

            # Get predictions from models
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

            # Calculate ensemble prediction
            over_votes = sum(1 for pred in predictions.values() if pred == 1)
            total_votes = len(predictions)
            over_percentage = over_votes / total_votes

            avg_over_prob = np.mean([prob['over_prob'] for prob in probabilities.values()])
            avg_under_prob = np.mean([prob['under_prob'] for prob in probabilities.values()])

            # Final recommendation
            if over_percentage > 0.5:
                recommendation = "OVER"
                confidence = avg_over_prob
            else:
                recommendation = "UNDER"
                confidence = avg_under_prob

            # Create result with matchup analysis
            result = {
                'player_name': player_name,
                'market_type': market_type,
                'prop_line': prop_line,
                'recommendation': recommendation,
                'confidence': confidence,
                'over_percentage': over_percentage,
                'models_agreement': f"{over_votes}/{total_votes}",
                'matchup_insights': {
                    'opponent': opponent_team,
                    'historical_avg_vs_opponent': matchup_history.get('avg_points_vs_opponent', 0),
                    'historical_over_rate_vs_opponent': matchup_history.get('over_rate_vs_opponent', 0.5),
                    'opponent_def_strength': opponent_defense.get('opp_def_opponent_def_strength', 0),
                    'matchup_games': matchup_history.get('matchup_games_vs_opponent', 0),
                    'matchup_advantage': matchup_features.get('matchup_advantage', 0)
                }
            }

            return result

        except Exception as e:
            print(f"❌ Error making enhanced prediction for {player_name}: {e}")
            return None

    def run_enhanced_predictions(self):
        """Run enhanced daily predictions with matchup analysis."""

        try:
            # Load models and data
            self.load_models_and_data()

            # Fetch today's player props
            print("\n=== FETCHING TODAY'S PLAYER PROPS ===")
            props_df = self.odds_client.get_all_todays_player_props()

            if props_df.empty:
                print("❌ No player props available for today")
                return None

            # Format props data
            formatted_props = self.odds_client.format_for_ml_pipeline(props_df)

            print(f"\n=== GENERATING ENHANCED PREDICTIONS FOR {len(formatted_props)} PROPS ===")

            predictions = []

            for _, prop in formatted_props.iterrows():
                print(f"\nAnalyzing: {prop['fullName']} {prop['market_type']} O/U {prop['prop_line']}")

                # Determine opponent team and player team
                # Since props don't have playerteamName, we'll infer from the game context
                # For now, assume player is on home team (simplified - in production you'd need player-to-team mapping)
                player_team = prop['home_team']  # This is a simplification
                opponent_team = prop['away_team'] if player_team == prop['home_team'] else prop['home_team']

                game_context = {
                    'game_date': prop['gameDate'],
                    'home_team': prop['home_team'],
                    'away_team': prop['away_team'],
                    'player_team': player_team,
                    'opponent_team': opponent_team
                }

                prediction = self.make_enhanced_prediction(
                    prop['fullName'],
                    prop['prop_line'],
                    prop['market_type'],
                    game_context
                )

                if prediction:
                    # Add odds and bookmaker info
                    prediction.update({
                        'over_odds': prop['over_odds'],
                        'bookmaker': prop['bookmaker'],
                        'game_time': prop['game_time']
                    })
                    predictions.append(prediction)

            if not predictions:
                print("❌ No predictions generated")
                return None

            # Create results DataFrame
            results_df = pd.DataFrame(predictions)

            # Filter for high-confidence predictions with matchup data
            print(f"\n=== FILTERING HIGH-CONFIDENCE PREDICTIONS ===")

            high_confidence = results_df[
                (results_df['confidence'] > 0.65) &
                (results_df['over_percentage'].isin([0.0, 1.0]))
            ].copy()

            # Sort by confidence
            high_confidence = high_confidence.sort_values('confidence', ascending=False)

            # Display results with matchup insights
            self.display_enhanced_predictions(high_confidence)

            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            output_file = f"../data/processed/enhanced_predictions_{timestamp}.csv"
            results_df.to_csv(output_file, index=False)

            print(f"\n✅ Saved enhanced predictions to: {output_file}")
            print(f"✅ Generated {len(results_df)} total predictions")
            print(f"✅ {len(high_confidence)} high-confidence recommendations")

            return high_confidence

        except Exception as e:
            print(f"❌ Error in enhanced predictions: {e}")
            return None

    def display_enhanced_predictions(self, predictions_df):
        """Display predictions with matchup analysis."""

        if predictions_df.empty:
            print("\n🤷 No high-confidence predictions for today")
            return

        print(f"\n🎯 ENHANCED BETTING RECOMMENDATIONS WITH MATCHUP ANALYSIS ({len(predictions_df)} total)")
        print("=" * 100)

        for _, pred in predictions_df.iterrows():
            print(f"\n🏀 {pred['player_name']} - {pred['market_type'].upper()}")
            print(f"   Line: {pred['prop_line']}")
            print(f"   Game: vs {pred['matchup_insights']['opponent']}")
            print(f"   📊 RECOMMENDATION: {pred['recommendation']}")
            print(f"   📈 Confidence: {pred['confidence']:.1%}")
            print(f"   🤝 Model Agreement: {pred['models_agreement']}")
            print(f"   💰 Odds: {pred['over_odds']:+}")
            print(f"   🏪 Book: {pred['bookmaker']}")

            # Matchup insights
            insights = pred['matchup_insights']
            print(f"   📈 MATCHUP INSIGHTS:")
            print(f"      • Historical avg vs opponent: {insights['historical_avg_vs_opponent']:.1f} pts")
            print(f"      • Historical over rate: {insights['historical_over_rate_vs_opponent']:.1%}")
            print(f"      • Opponent defense: {'STRONG' if insights['opponent_def_strength'] > 0.1 else 'WEAK' if insights['opponent_def_strength'] < -0.1 else 'AVERAGE'}")
            print(f"      • Matchup advantage: {insights['matchup_advantage']:+.1f} pts")
            if insights['matchup_games'] > 0:
                print(f"      • Sample size: {insights['matchup_games']} games")

        print("\n" + "=" * 100)
        print("⚠️  DISCLAIMER: Enhanced model considers matchup history. Use for educational purposes only.")
        print("    Always do your own research and gamble responsibly.")

def main():
    """Main function to run enhanced daily predictions."""

    # Check for API key
    if not os.getenv('ODDS_API_KEY'):
        print("❌ Missing API key!")
        print("\nTo use this system:")
        print("1. Get API key from https://the-odds-api.com/")
        print("2. Set environment variable:")
        print("   export ODDS_API_KEY='your_api_key_here'")
        print("\nThen run: python src/enhanced_daily_predictions.py")
        return

    try:
        # Initialize and run enhanced predictor
        predictor = EnhancedDailyPredictor()
        predictions = predictor.run_enhanced_predictions()

        if predictions is not None and not predictions.empty:
            print(f"\n🎉 Enhanced daily predictions complete! Found {len(predictions)} high-confidence bets.")
        else:
            print("\n📊 Daily analysis complete. No high-confidence predictions today.")

    except Exception as e:
        print(f"\n💥 Error running enhanced predictions: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()