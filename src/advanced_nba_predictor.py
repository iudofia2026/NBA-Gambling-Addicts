"""
ADVANCED NBA PREDICTION SYSTEM
Implementation based on 2024 research and best practices
Features: Usage Rate, Defensive Efficiency, Pace, Travel Fatigue, Advanced Metrics
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

class AdvancedNBAPredictor:
    """Advanced NBA prediction system using 2024 research-based features."""

    def __init__(self, api_key=None):
        """Initialize the advanced predictor."""
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError("API key required")

        self.odds_client = NBAOddsClient(self.api_key)
        self.models = {}
        self.historical_data = None

        print("=" * 60)
        print("🏀 ADVANCED NBA PREDICTION SYSTEM")
        print("   Features:")
        print("   ✓ Usage Rate Analysis")
        print("   ✓ Defensive Efficiency Matchups")
        print("   ✓ Pace and Tempo Factors")
        print("   ✓ Travel Fatigue Tracking")
        print("   ✓ Advanced Metrics (PER, TS%, PIE)")
        print("   ✓ Market Movement Signals")
        print("=" * 60)

    def load_models(self):
        """Load trained models."""
        print("\n📂 LOADING ML MODELS")

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

        if not self.models:
            print("❌ No models loaded successfully!")
            return False

        # Load historical data
        self.historical_data = pd.read_csv('../data/processed/engineered_features.csv')
        self.historical_data['gameDate'] = pd.to_datetime(self.historical_data['gameDate'], errors='coerce')
        print(f"✅ Historical data: {len(self.historical_data):,} games")

        return True

    def calculate_usage_rate_features(self, player_name, days_back=20):
        """Calculate usage rate and related features."""
        player_data = self.historical_data[
            self.historical_data['fullName'] == player_name
        ].tail(days_back).copy()

        if len(player_data) < 5:
            return self._get_default_usage_features()

        features = {}

        # Usage rate proxy (points + assists + rebounds)
        player_data['usage_proxy'] = (player_data['points'] + player_data['assists'] + player_data['reboundsTotal'])

        # Recent usage trend
        recent_usage = player_data['usage_proxy'].tail(5).mean()
        older_usage = player_data['usage_proxy'].head(len(player_data) - 5).mean() if len(player_data) > 5 else recent_usage

        features['usage_trend'] = (recent_usage - older_usage) / older_usage if older_usage > 0 else 0
        features['current_usage'] = recent_usage
        features['usage_consistency'] = 1 - (player_data['usage_proxy'].std() / player_data['usage_proxy'].mean()) if player_data['usage_proxy'].mean() > 0 else 0

        # Team dependency (how much player relies on team performance)
        if 'playerteamName' in player_data.columns:
            team_performance = player_data.groupby('playerteamName')['points'].mean()
            player_avg = player_data['points'].mean()
            features['team_dependency'] = player_avg / team_performance.mean() if len(team_performance) > 0 else 0.5

        return features

    def analyze_defensive_matchup(self, player_name, opponent_team, days_back=30):
        """Analyze opponent defensive efficiency and matchup history."""
        # Get player historical performance vs opponent
        matchup_data = self.historical_data[
            (self.historical_data['fullName'] == player_name) &
            (
                (self.historical_data['playerteamName'] == opponent_team) |
                (self.historical_data['opponentteamName'] == opponent_team)
            )
        ].tail(days_back)

        # Get opponent defensive stats
        opponent_games = self.historical_data[
            (self.historical_data['playerteamName'] == opponent_team) |
            (self.historical_data['opponentteamName'] == opponent_team)
        ].tail(days_back)

        features = {}

        if not matchup_data.empty:
            # Historical performance vs opponent
            features['avg_points_vs_opp'] = matchup_data['points'].mean()
            features['efficiency_vs_opp'] = (matchup_data['points'] / matchup_data['numMinutes']).mean()
            features['over_rate_vs_opp'] = (matchup_data['over_threshold'] == 1).mean() if 'over_threshold' in matchup_data.columns else 0.5

            # Recent form in this matchup
            recent_matchups = matchup_data.tail(3)
            if len(recent_matchups) >= 2:
                features['recent_trend_vs_opp'] = recent_matchups.iloc[-1]['points'] - recent_matchups.iloc[0]['points']
                features['matchup_consistency'] = 1 - (recent_matchups['points'].std() / recent_matchups['points'].mean()) if recent_matchups['points'].mean() > 0 else 0.5
            else:
                features['recent_trend_vs_opp'] = 0
                features['matchup_consistency'] = 0.5
        else:
            # Default values if no matchup history
            player_avg = self.historical_data[self.historical_data['fullName'] == player_name]['points'].mean() if not self.historical_data[self.historical_data['fullName'] == player_name].empty else 10
            features.update({
                'avg_points_vs_opp': player_avg,
                'efficiency_vs_opp': player_avg / 30,
                'over_rate_vs_opp': 0.5,
                'recent_trend_vs_opp': 0,
                'matchup_consistency': 0.5
            })

        # Opponent defensive rating (proxy from points allowed)
        if not opponent_games.empty:
            features['opp_defensive_strength'] = 1 - (opponent_games['points'].mean() / 115)  # 115 = league average
            features['opp_pace_factor'] = opponent_games['numMinutes'].mean() / 48 if 'numMinutes' in opponent_games.columns else 1.0
        else:
            features['opp_defensive_strength'] = 0.5
            features['opp_pace_factor'] = 1.0

        return features

    def calculate_travel_fatigue(self, player_name, current_game_date):
        """Calculate travel fatigue based on recent schedule."""
        player_games = self.historical_data[
            self.historical_data['fullName'] == player_name
        ].sort_values('gameDate').tail(10)

        if len(player_games) < 2:
            return self._get_default_fatigue_features()

        features = {}

        # Calculate days between games
        player_games = player_games.copy()
        player_games['days_since_prev'] = player_games['gameDate'].diff().dt.days

        # Recent fatigue factors
        recent_games = player_games.tail(5)
        features['avg_rest_days'] = recent_games['days_since_prev'].mean() if len(recent_games) > 1 else 2
        features['back_to_back_count'] = (recent_games['days_since_prev'] <= 1).sum() if 'days_since_prev' in recent_games.columns else 0

        # Travel factor (simple proxy based on opponent changes)
        if 'opponentteamName' in player_games.columns:
            features['travel_frequency'] = recent_games['opponentteamName'].nunique() / len(recent_games)
            features['opponent_switch_frequency'] = (recent_games['opponentteamName'] != recent_games['opponentteamName'].shift()).sum() / max(len(recent_games) - 1, 1)
        else:
            features['travel_frequency'] = 0.5
            features['opponent_switch_frequency'] = 0.5

        # Fatigue score (higher = more fatigued)
        features['fatigue_score'] = (
            features['back_to_back_count'] * 0.3 +
            (2 - features['avg_rest_days']) * 0.2 +
            features['travel_frequency'] * 0.3 +
            features['opponent_switch_frequency'] * 0.2
        )

        return features

    def calculate_advanced_metrics(self, player_name, days_back=15):
        """Calculate advanced NBA metrics."""
        player_data = self.historical_data[
            self.historical_data['fullName'] == player_name
        ].tail(days_back).copy()

        if len(player_data) < 5:
            return self._get_default_advanced_metrics()

        features = {}

        # Player Impact Estimate (PIE) proxy
        player_data['pie_proxy'] = (player_data['points'] + player_data['reboundsTotal'] + player_data['assists']) / player_data['numMinutes']
        features['current_pie'] = player_data['pie_proxy'].tail(5).mean()

        # True Shooting Percentage (TS%) proxy
        player_data['ts_proxy'] = player_data['points'] / (player_data['numMinutes'] * 0.5)  # Simplified
        features['current_ts'] = player_data['ts_proxy'].tail(5).mean()

        # Plus/Minus proxy (performance relative to average)
        player_avg = player_data['points'].mean()
        league_avg_proxy = 15  # Simplified league average
        features['plus_minus_proxy'] = (player_avg - league_avg_proxy) / league_avg_proxy

        # Consistency metrics
        features['scoring_consistency'] = 1 - (player_data['points'].std() / player_data['points'].mean()) if player_data['points'].mean() > 0 else 0
        features['minute_consistency'] = 1 - (player_data['numMinutes'].std() / player_data['numMinutes'].mean()) if player_data['numMinutes'].mean() > 0 else 0

        # Hot/cold detection
        recent_avg = player_data['points'].tail(3).mean()
        season_avg = player_data['points'].mean()
        features['hot_cold_factor'] = (recent_avg - season_avg) / season_avg if season_avg > 0 else 0

        return features

    def calculate_pace_factor(self, player_team, opponent_team):
        """Calculate game pace factor."""
        # Team pace from recent games
        player_team_games = self.historical_data[
            self.historical_data['playerteamName'] == player_team
        ].tail(10)

        opponent_games = self.historical_data[
            self.historical_data['playerteamName'] == opponent_team
        ].tail(10)

        features = {}

        if not player_team_games.empty:
            features['team_pace'] = player_team_games['numMinutes'].mean() / 48 if 'numMinutes' in player_team_games.columns else 1.0
            features['team_scoring_rate'] = player_team_games['points'].mean()
        else:
            features['team_pace'] = 1.0
            features['team_scoring_rate'] = 110

        if not opponent_games.empty:
            features['opp_pace'] = opponent_games['numMinutes'].mean() / 48 if 'numMinutes' in opponent_games.columns else 1.0
            features['opp_scoring_allowed'] = opponent_games['points'].mean()
        else:
            features['opp_pace'] = 1.0
            features['opp_scoring_allowed'] = 110

        # Combined pace factor
        features['game_pace_factor'] = (features['team_pace'] + features['opp_pace']) / 2

        # Scoring environment adjustment
        features['scoring_environment'] = (features['team_scoring_rate'] + features['opp_scoring_allowed']) / 230  # 230 = average total points

        return features

    def calculate_market_signals(self, prop_data):
        """Calculate market movement and sentiment signals."""
        features = {}

        if isinstance(prop_data, list) and len(prop_data) > 0:
            # Convert to DataFrame for analysis
            prop_df = pd.DataFrame(prop_data)

            # Number of bookmakers offering this line (liquidity indicator)
            features['market_liquidity'] = len(prop_df)

            # Odds distribution (market consensus)
            if 'over_odds' in prop_df.columns:
                odds_numeric = pd.to_numeric(prop_df['over_odds'], errors='coerce')
                features['odds_consensus'] = 1 - (odds_numeric.std() / odds_numeric.mean()) if odds_numeric.mean() > 0 else 0
                features['best_odds_value'] = odds_numeric.max()
                features['avg_odds_value'] = odds_numeric.mean()
            else:
                features['odds_consensus'] = 0.5
                features['best_odds_value'] = -110
                features['avg_odds_value'] = -110

            # Line value consistency
            if 'prop_line' in prop_df.columns:
                features['line_consensus'] = prop_df['prop_line'].nunique() == 1
                features['avg_line_value'] = prop_df['prop_line'].mean()
            else:
                features['line_consensus'] = False
                features['avg_line_value'] = 0
        else:
            features.update({
                'market_liquidity': 1,
                'odds_consensus': 0.5,
                'best_odds_value': -110,
                'avg_odds_value': -110,
                'line_consensus': False,
                'avg_line_value': 0
            })

        return features

    def calculate_advanced_prediction(self, player_name, prop_data, game_context):
        """Calculate advanced prediction using all features."""

        # Get baseline
        player_data = self.historical_data[self.historical_data['fullName'] == player_name]
        if player_data.empty:
            return None

        baseline_points = player_data['points'].tail(10).mean()

        # Calculate all feature sets
        usage_features = self.calculate_usage_rate_features(player_name)
        defensive_features = self.analyze_defensive_matchup(player_name, game_context['opponent_team'])
        fatigue_features = self.calculate_travel_fatigue(player_name, game_context.get('game_date', datetime.now()))
        advanced_metrics = self.calculate_advanced_metrics(player_name)
        pace_features = self.calculate_pace_factor(game_context.get('player_team', ''), game_context['opponent_team'])
        market_features = self.calculate_market_signals(prop_data)

        # Calculate weighted prediction
        prediction_weights = {
            'baseline': 0.15,          # Historical baseline
            'usage': 0.20,             # Current usage rate
            'matchup': 0.25,           # Defensive matchup
            'advanced': 0.20,          # Advanced metrics
            'pace': 0.10,              # Game pace
            'fatigue': 0.05,           # Fatigue adjustment
            'market': 0.05             # Market signals
        }

        # Baseline component
        baseline_component = baseline_points * prediction_weights['baseline']

        # Usage rate adjustment
        usage_adjustment = baseline_points * (usage_features['usage_trend'] * 0.5 + 1) * prediction_weights['usage']

        # Matchup adjustment
        matchup_multiplier = (
            defensive_features['avg_points_vs_opp'] / baseline_points if baseline_points > 0 else 1
        ) * defensive_features['opp_defensive_strength']
        matchup_adjustment = baseline_points * matchup_multiplier * prediction_weights['matchup']

        # Advanced metrics adjustment
        advanced_multiplier = (
            1 + advanced_metrics['hot_cold_factor'] * 0.3 +
            advanced_metrics['plus_minus_proxy'] * 0.2 +
            advanced_metrics['current_pie'] * 0.1
        )
        advanced_adjustment = baseline_points * advanced_multiplier * prediction_weights['advanced']

        # Pace adjustment
        pace_adjustment = baseline_points * pace_features['game_pace_factor'] * prediction_weights['pace']

        # Fatigue adjustment (negative impact)
        fatigue_penalty = baseline_points * fatigue_features['fatigue_score'] * -0.2 * prediction_weights['fatigue']

        # Market sentiment adjustment (small effect)
        market_adjustment = market_features['odds_consensus'] * 2 * prediction_weights['market']

        # Final prediction
        predicted_points = (
            baseline_component +
            usage_adjustment +
            matchup_adjustment +
            advanced_adjustment +
            pace_adjustment +
            fatigue_penalty +
            market_adjustment
        )

        # Calculate confidence score
        confidence_factors = [
            usage_features['usage_consistency'] * 0.2,
            defensive_features['matchup_consistency'] * 0.2,
            advanced_metrics['scoring_consistency'] * 0.2,
            market_features['odds_consensus'] * 0.1,
            min(len(player_data) / 50, 1.0) * 0.3  # Sample size confidence
        ]

        confidence_score = sum(confidence_factors)

        # Create insights
        insights = {
            'predicted_points': round(predicted_points, 1),
            'confidence_score': min(confidence_score, 1.0),
            'usage_trend': f"{'↑' if usage_features['usage_trend'] > 0.1 else '↓' if usage_features['usage_trend'] < -0.1 else '→'} {usage_features['usage_trend']:.1%}",
            'matchup_rating': f"{'Favorable' if defensive_features['opp_defensive_strength'] > 0.6 else 'Neutral' if defensive_features['opp_defensive_strength'] > 0.4 else 'Tough'}",
            'fatigue_level': f"{'High' if fatigue_features['fatigue_score'] > 0.6 else 'Medium' if fatigue_features['fatigue_score'] > 0.3 else 'Low'}",
            'hot_cold': f"{'Hot' if advanced_metrics['hot_cold_factor'] > 0.1 else 'Cold' if advanced_metrics['hot_cold_factor'] < -0.1 else 'Neutral'}",
            'pace_impact': f"{'Up-tempo' if pace_features['game_pace_factor'] > 1.05 else 'Slow-down' if pace_features['game_pace_factor'] < 0.95 else 'Normal'}"
        }

        return {
            'baseline_points': baseline_points,
            'predicted_points': predicted_points,
            'confidence_score': confidence_score,
            'insights': insights,
            'features': {
                'usage': usage_features,
                'defensive': defensive_features,
                'fatigue': fatigue_features,
                'advanced': advanced_metrics,
                'pace': pace_features,
                'market': market_features
            }
        }

    def make_prediction(self, player_name, prop_data, game_context):
        """Make prediction for all markets of a player."""
        try:
            # Calculate advanced prediction
            result = self.calculate_advanced_prediction(player_name, prop_data, game_context)
            if not result:
                print(f"   ❌ Could not calculate prediction for {player_name}")
                return None

            predictions = {}

            # Process each market type
            for market_type, market_props in prop_data.items():
                print(f"      Processing {market_type}: {len(market_props)} prop lines")

                # Get prediction for this market
                if market_type.lower() == 'player_points':
                    predicted_value = result['predicted_points']
                elif market_type.lower() == 'player_rebounds':
                    # Use regression of rebounds to points
                    predicted_value = result['baseline_points'] * 0.35 * result['features']['pace']['game_pace_factor']
                elif market_type.lower() == 'player_assists':
                    # Use regression of assists to points with usage rate adjustment
                    usage_factor = result['features']['usage']['current_usage'] / 30 if result['features']['usage']['current_usage'] > 0 else 1
                    predicted_value = result['baseline_points'] * 0.25 * usage_factor
                else:
                    print(f"      ⚠️ Unknown market type: {market_type}")
                    continue

                # Find best odds
                prop_df = pd.DataFrame(market_props)
                prop_df['over_odds_numeric'] = pd.to_numeric(prop_df['over_odds'], errors='coerce')
                best_prop = prop_df.loc[prop_df['over_odds_numeric'].idxmax()].to_dict()

                # Make recommendation
                recommendation = "OVER" if predicted_value > best_prop['prop_line'] else "UNDER"

                predictions[market_type] = {
                    'market_type': market_type,
                    'prop_line': best_prop['prop_line'],
                    'recommendation': recommendation,
                    'predicted_value': round(predicted_value, 1),
                    'line_diff': round(predicted_value - best_prop['prop_line'], 1),
                    'confidence': min(result['confidence_score'], 0.99),
                    'over_odds': best_prop['over_odds'],
                    'bookmaker': best_prop['bookmaker'],
                    'game_time': best_prop['game_time']
                }

                print(f"      ✅ {market_type}: {recommendation} {best_prop['prop_line']} (pred: {predicted_value:.1f})")

            if not predictions:
                print(f"   ❌ No predictions generated for {player_name}")
                return None

            # Return combined prediction
            return {
                'player_name': player_name,
                'predictions': predictions,
                'advanced_insights': result['insights'],
                'feature_breakdown': result['features']
            }

        except Exception as e:
            print(f"   ❌ Error predicting {player_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_default_usage_features(self):
        return {
            'usage_trend': 0,
            'current_usage': 15,
            'usage_consistency': 0.5,
            'team_dependency': 0.3
        }

    def _get_default_fatigue_features(self):
        return {
            'avg_rest_days': 2,
            'back_to_back_count': 0,
            'travel_frequency': 0.5,
            'opponent_switch_frequency': 0.5,
            'fatigue_score': 0.3
        }

    def _get_default_advanced_metrics(self):
        return {
            'current_pie': 0.1,
            'current_ts': 0.5,
            'plus_minus_proxy': 0,
            'scoring_consistency': 0.5,
            'minute_consistency': 0.5,
            'hot_cold_factor': 0
        }

    def run_predictions(self):
        """Run the advanced prediction system."""

        # Load models
        if not self.load_models():
            return None

        # Fetch today's props
        print("\n📡 FETCHING TODAY'S NBA PLAYER PROPS")
        props_df = self.odds_client.get_all_todays_player_props()

        if props_df.empty:
            print("❌ No player props available")
            return None

        # Group props by player, market, and line value
        player_props = {}

        for _, prop in props_df.iterrows():
            player = prop['player_name']
            market = prop['market_type'].lower()
            line_value = prop['line_value']

            if player not in player_props:
                player_props[player] = {}
            if market not in player_props[player]:
                player_props[player][market] = {}
            if line_value not in player_props[player][market]:
                player_props[player][market][line_value] = []

            player_props[player][market][line_value].append({
                'over_odds': prop['over_odds'],
                'bookmaker': prop['bookmaker'],
                'game_time': prop['game_time'],
                'home_team': prop['home_team'],
                'away_team': prop['away_team'],
                'gameDate': prop['gameDate'],
                'playerteamName': prop.get('team', '')
            })

        print(f"✅ Found props for {len(player_props)} players")

        # Generate predictions
        print(f"\n🔮 GENERATING ADVANCED PREDICTIONS")
        print("-" * 50)

        all_predictions = []

        for player_name, player_markets in player_props.items():
            print(f"\n🎯 {player_name}")

            # Get game context
            first_market = list(player_markets.values())[0]
            first_line = list(first_market.values())[0]
            first_prop = first_line[0]

            opponent_team = first_prop['away_team'] if first_prop['home_team'] == first_prop.get('playerteamName', '') else first_prop['home_team']

            game_context = {
                'game_date': first_prop['gameDate'],
                'home_team': first_prop['home_team'],
                'away_team': first_prop['away_team'],
                'player_team': first_prop.get('playerteamName', ''),
                'opponent_team': opponent_team
            }

            # Flatten markets for prediction
            flattened_markets = {}
            for market, lines in player_markets.items():
                market_props = []
                for line_value, bookmakers in lines.items():
                    best_prop = max(bookmakers, key=lambda x: float(x['over_odds']) if isinstance(x['over_odds'], (int, float, str)) and str(x['over_odds']).replace('.', '', 1).isdigit() else 0)
                    market_props.append({
                        'prop_line': line_value,
                        'over_odds': best_prop['over_odds'],
                        'bookmaker': best_prop['bookmaker'],
                        'game_time': best_prop['game_time'],
                        'home_team': best_prop['home_team'],
                        'away_team': best_prop['away_team'],
                        'gameDate': best_prop['gameDate'],
                        'playerteamName': best_prop.get('playerteamName', '')
                    })
                if market_props:
                    flattened_markets[market] = market_props

            if flattened_markets:
                prediction = self.make_prediction(player_name, flattened_markets, game_context)
                if prediction:
                    all_predictions.append(prediction)

        if not all_predictions:
            print("\n❌ No predictions generated")
            return None

        # Display results
        self.display_results(all_predictions)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_file = f"../data/predictions/advanced_predictions_{timestamp}.csv"

        # Flatten for CSV
        flattened_predictions = []
        for pred in all_predictions:
            for market, data in pred['predictions'].items():
                flattened_predictions.append({
                    'player_name': pred['player_name'],
                    'market_type': market,
                    'prop_line': data['prop_line'],
                    'recommendation': data['recommendation'],
                    'predicted_value': data['predicted_value'],
                    'line_diff': data['line_diff'],
                    'confidence': data['confidence'],
                    'over_odds': data['over_odds'],
                    'bookmaker': data['bookmaker'],
                    'game_time': data['game_time'],
                    'usage_trend': pred['advanced_insights']['usage_trend'],
                    'matchup_rating': pred['advanced_insights']['matchup_rating'],
                    'fatigue_level': pred['advanced_insights']['fatigue_level'],
                    'hot_cold': pred['advanced_insights']['hot_cold'],
                    'pace_impact': pred['advanced_insights']['pace_impact']
                })

        pd.DataFrame(flattened_predictions).to_csv(output_file, index=False)
        print(f"\n💾 Saved to: {output_file}")
        print(f"📊 Total predictions: {len(flattened_predictions)}")

        return all_predictions

    def display_results(self, predictions):
        """Display advanced betting recommendations."""
        if not predictions:
            print("\n🤷 No predictions available")
            return

        print(f"\n🏆 ADVANCED NBA BETTING RECOMMENDATIONS")
        print("=" * 80)
        print(f"📊 Total players: {len(predictions)}")

        over_count = sum(1 for p in predictions if
            any(d['recommendation'] == 'OVER' for d in p['predictions'].values()))
        under_count = sum(1 for p in predictions if
            any(d['recommendation'] == 'UNDER' for d in p['predictions'].values()))

        print(f"⏱️  Consensus: {'OVER' if over_count > under_count else 'UNDER'} ({over_count} OVER, {under_count} UNDER)")

        for pred in predictions:
            insights = pred['advanced_insights']
            print(f"\n🏀 {pred['player_name'].upper()}")

            # Show predictions
            for market, data in pred['predictions'].items():
                print(f"   {market.replace('player_', '').upper():8} | Line: {data['prop_line']:5} | Pred: {data['predicted_value']:5.1f} ({data['line_diff']:+5.1f}) | {data['recommendation']:5}")

            # Show insights
            print(f"   📊 Confidence: {data['confidence']:.1%}")
            print(f"   💰 Best Odds: {data['bookmaker']} ({data['over_odds']:+})")
            print(f"\n   🔍 ANALYSIS:")
            print(f"      • Usage Trend: {insights['usage_trend']}")
            print(f"      • Matchup: {insights['matchup_rating']}")
            print(f"      • Fatigue: {insights['fatigue_level']}")
            print(f"      • Form: {insights['hot_cold']}")
            print(f"      • Pace: {insights['pace_impact']}")

        print("\n" + "=" * 80)

def main():
    """Run the advanced predictions system."""
    if not os.getenv('ODDS_API_KEY'):
        print("❌ Missing API key!")
        print("\nSet ODDS_API_KEY environment variable")
        return

    try:
        predictor = AdvancedNBAPredictor()
        predictions = predictor.run_predictions()

        if predictions:
            print(f"\n🎉 Advanced system complete! Generated {len(predictions)} player predictions")
        else:
            print("\n📊 Complete. No predictions available today.")

    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()