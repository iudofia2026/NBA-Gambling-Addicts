"""
FIXED NBA PREDICTIONS SYSTEM
Fixed to show each player once with best odds from all bookmakers
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

class FixedNBAPredictionsSystem:
    """Fixed NBA predictions system that shows each player once with best odds."""

    def __init__(self, api_key=None):
        """Initialize the fixed predictions system."""
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError("API key required")

        self.odds_client = NBAOddsClient(self.api_key)
        self.models = {}
        self.historical_data = None

        print("=" * 60)
        print("🏀 NBA PREDICTIONS SYSTEM - FIXED VERSION")
        print("   Each player shown once with best odds")
        print("   All 3 iterations integrated")
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

    def analyze_player_form(self, player_name, days_back=15):
        """Analyze player's current form."""
        player_data = self.historical_data[
            self.historical_data['fullName'] == player_name
        ].tail(days_back).copy()

        if player_data.empty:
            return self._get_default_form()

        recent_games = player_data.tail(5)

        momentum = {}

        if len(recent_games) >= 3:
            momentum['current_streak'] = self._detect_streak(recent_games['points'])
            momentum['streak_length'] = len(recent_games)
            momentum['streak_intensity'] = np.std(recent_games['points']) / recent_games['points'].mean() if recent_games['points'].mean() > 0 else 0
        else:
            momentum['current_streak'] = 'neutral'
            momentum['streak_length'] = len(recent_games)
            momentum['streak_intensity'] = 0

        if len(recent_games) >= 5:
            momentum['trend'] = np.polyfit(range(len(recent_games)), recent_games['points'], 1)[0]
            momentum['trend_strength'] = abs(momentum['trend'])
        else:
            momentum['trend'] = 0
            momentum['trend_strength'] = 0

        momentum['form_confidence'] = 1 - (player_data['points'].std() / player_data['points'].mean() if player_data['points'].mean() > 0 else 0.5)

        return momentum

    def analyze_team_chemistry(self, player_name, player_team, days_back=10):
        """Analyze team chemistry and lineup impact."""
        team_games = self.historical_data[
            (self.historical_data['playerteamName'] == player_team) |
            (self.historical_data['opponentteamName'] == player_team)
        ].tail(days_back)

        if team_games.empty:
            return self._get_default_chemistry()

        chemistry = {}

        recent_team_performance = team_games.tail(5)['points'].mean()
        older_team_performance = team_games.head(len(team_games) - 5)['points'].mean() if len(team_games) > 5 else recent_team_performance

        chemistry['team_momentum'] = recent_team_performance - older_team_performance
        chemistry['momentum_consistency'] = 1 - (team_games.tail(5)['points'].std() / team_games.tail(5)['points'].mean() if team_games.tail(5)['points'].mean() > 0 else 0.5)

        player_team_games = team_games[team_games['fullName'] == player_name]
        if len(player_team_games) > 0:
            chemistry['team_usage_share'] = player_team_games['points'].mean() / recent_team_performance
            chemistry['chemistry_impact'] = min(chemistry['team_usage_share'] * chemistry['momentum_consistency'], 1.0)
        else:
            chemistry['team_usage_share'] = 0.2
            chemistry['chemistry_impact'] = 0.5

        return chemistry

    def analyze_individual_matchup(self, player_name, opponent_team, days_back=30):
        """Analyze individual player vs team matchups."""
        matchup_data = self.historical_data[
            (self.historical_data['fullName'] == player_name) &
            (
                (self.historical_data['playerteamName'] == opponent_team) |
                (self.historical_data['opponentteamName'] == opponent_team)
            )
        ].tail(days_back)

        if matchup_data.empty:
            return self._get_default_matchup()

        matchup = {}

        matchup['avg_points_vs_opp'] = matchup_data['points'].mean()
        matchup['over_rate_vs_opp'] = (matchup_data['over_threshold'] == 1).mean()
        matchup['efficiency_vs_opp'] = (matchup_data['points'] / matchup_data['numMinutes']).mean()

        recent_matchups = matchup_data.tail(3)
        if len(recent_matchups) >= 2:
            matchup['recent_trend_vs_opp'] = recent_matchups.tail(1)['points'].iloc[0] - recent_matchups.head(1)['points'].iloc[0]
            matchup['consistency_vs_opp'] = 1 - (recent_matchups['points'].std() / recent_matchups['points'].mean())
        else:
            matchup['recent_trend_vs_opp'] = 0
            matchup['consistency_vs_opp'] = 0.5

        matchup['sample_confidence'] = min(len(matchup_data) / 10, 1.0)

        return matchup

    def calculate_final_prediction(self, player_name, game_context, form, chemistry, matchup, baseline_points):
        """Calculate final prediction with all factors weighted."""

        form_weight = 0.35
        chemistry_weight = 0.20
        matchup_weight = 0.25
        baseline_weight = 0.20

        form_adjusted = form['form_confidence'] * form_weight
        chemistry_adjusted = chemistry['chemistry_impact'] * chemistry_weight
        matchup_adjusted = matchup['sample_confidence'] * matchup_weight

        form_component = baseline_points * (1 + (form['trend'] / baseline_points if baseline_points > 0 else 0))
        chemistry_component = chemistry_adjusted * 5
        matchup_component = (matchup['avg_points_vs_opp'] - baseline_points) * matchup_adjusted

        predicted_points = (
            form_component * form_weight +
            chemistry_component +
            matchup_component +
            baseline_points * baseline_weight
        )

        confidence_score = (
            form['form_confidence'] * 0.3 +
            chemistry['momentum_consistency'] * 0.2 +
            matchup['consistency_vs_opp'] * 0.2 +
            0.3
        )
        confidence_score = min(confidence_score, 1.0)

        return {
            'predicted_points': predicted_points,
            'confidence_score': min(confidence_score, 1.0),
            'form_factor': form,
            'chemistry_factor': chemistry,
            'matchup_factor': matchup
        }

    def make_prediction(self, player_name, prop_lines_by_market, game_context):
        """Make final prediction for all markets of a player."""
        try:
            # Get baseline from historical data
            player_data = self.historical_data[self.historical_data['fullName'] == player_name]
            if player_data.empty:
                return None

            baseline_points = player_data['points'].tail(10).mean()

            # Analyze all three iterations
            form = self.analyze_player_form(player_name)
            chemistry = self.analyze_team_chemistry(player_name, game_context.get('player_team', ''))
            matchup = self.analyze_individual_matchup(player_name, game_context['opponent_team'])

            # Calculate final prediction
            result = self.calculate_final_prediction(
                player_name, game_context, form, chemistry, matchup, baseline_points
            )

            predictions = {}

            # Make prediction for each market type
            for market_type, prop_data in prop_lines_by_market.items():
                if market_type.lower() == 'points':
                    predicted_value = result['predicted_points']
                elif market_type.lower() == 'rebounds':
                    predicted_value = baseline_points * 0.4
                elif market_type.lower() == 'assists':
                    predicted_value = baseline_points * 0.25
                else:
                    continue  # Skip unknown market types

                # Find best odds for this market
                best_prop = prop_data.sort_values('over_odds', key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else 0, ascending=False).iloc[0]

                recommendation = "OVER" if predicted_value > best_prop['prop_line'] else "UNDER"

                predictions[market_type] = {
                    'market_type': market_type,
                    'prop_line': best_prop['prop_line'],
                    'recommendation': recommendation,
                    'predicted_value': predicted_value,
                    'line_diff': round(predicted_value - best_prop['prop_line'], 1),
                    'confidence': min(result['confidence_score'], 0.99),
                    'over_odds': best_prop['over_odds'],
                    'bookmaker': best_prop['bookmaker'],
                    'game_time': best_prop['game_time']
                }

            if not predictions:
                return None

            # Return combined prediction
            return {
                'player_name': player_name,
                'predictions': predictions,
                'final_insights': {
                    'confidence_level': self._get_confidence_level(result['confidence_score']),
                    'form_analysis': f"{form['current_streak']} streak ({form['streak_length']} games)",
                    'team_chemistry': f"{'Improving' if chemistry['team_momentum'] > 0 else 'Neutral'}",
                    'matchup_history': f"{matchup['avg_points_vs_opp']:.1f} pts avg vs opponent"
                }
            }

        except Exception as e:
            print(f"❌ Error in prediction for {player_name}: {e}")
            return None

    def _get_confidence_level(self, confidence):
        """Convert confidence score to descriptive level."""
        if confidence >= 0.85:
            return "High"
        elif confidence >= 0.75:
            return "Medium-High"
        elif confidence >= 0.65:
            return "Medium"
        else:
            return "Low"

    def _detect_streak(self, points_series):
        """Detect if player is on a hot or cold streak."""
        if len(points_series) < 2:
            return 'neutral'

        recent_avg = points_series.tail(2).mean()
        historical_avg = points_series.mean()

        if recent_avg > historical_avg * 1.2:
            return 'hot'
        elif recent_avg < historical_avg * 0.8:
            return 'cold'
        else:
            return 'neutral'

    def _get_default_form(self):
        return {
            'current_streak': 'neutral',
            'streak_length': 0,
            'streak_intensity': 0,
            'trend': 0,
            'trend_strength': 0,
            'form_confidence': 0.5
        }

    def _get_default_chemistry(self):
        return {
            'team_momentum': 0,
            'momentum_consistency': 0.5,
            'team_usage_share': 0.2,
            'chemistry_impact': 0.5
        }

    def _get_default_matchup(self):
        return {
            'avg_points_vs_opp': 0,
            'over_rate_vs_opp': 0.5,
            'efficiency_vs_opp': 0,
            'recent_trend_vs_opp': 0,
            'consistency_vs_opp': 0.5,
            'sample_confidence': 0
        }

    def run_final_predictions(self):
        """Run the fixed predictions system."""

        # Load models
        if not self.load_models():
            return None

        # Fetch today's props
        print("\n📡 FETCHING TODAY'S NBA PLAYER PROPS")
        props_df = self.odds_client.get_all_todays_player_props()

        if props_df.empty:
            print("❌ No player props available")
            return None

        # Group props by player, market, AND line value to identify true duplicates
        player_props = {}
        line_cache = {}  # Cache to avoid re-running model for same lines

        for _, prop in props_df.iterrows():
            player = prop['player_name']
            market = prop['market_type'].lower()
            line_value = prop['line_value']

            # Create unique key for player-market-line combination
            unique_key = f"{player}_{market}_{line_value}"

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

        # Calculate statistics
        total_players = len(player_props)
        total_markets = sum(len(m) for m in player_props.values())
        total_unique_lines = sum(len(m) for p in player_props.values() for m in p.values())

        print(f"✅ Found props for {total_players} players")
        print(f"📊 Total market types: {total_markets}")
        print(f"🎯 Unique lines (deduplicated): {total_unique_lines}")
        print(f"💰 Avg bookmakers per line: {sum(len(bookmakers) for p in player_props.values() for m in p.values() for bookmakers in m.values()) / total_unique_lines:.1f}")

        # Generate predictions
        print(f"\n🔮 GENERATING FINAL PREDICTIONS")
        print(f"   Processing each unique line once with best odds")
        print("-" * 50)

        all_predictions = []

        for player_name, player_markets in player_props.items():
            print(f"\n🎯 Analyzing: {player_name}")

            # Flatten player's prop data for prediction (group all markets for this player)
            flattened_prop_data = {}

            # Get game context from first prop
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

            # Group markets with best odds (flatten the line structure)
            for market, lines in player_markets.items():
                # Convert lines dict to list format expected by make_prediction
                market_props = []
                for line_value, bookmakers in lines.items():
                    # Find best odds for this line
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

                if market_props:  # Only add if we have props for this market
                    flattened_prop_data[market] = market_props

            if flattened_prop_data:
                prediction = self.make_prediction(
                    player_name,
                    flattened_prop_data,
                    game_context
                )

                if prediction:
                    all_predictions.append(prediction)

        if not all_predictions:
            print("\n❌ No high-confidence predictions generated")
            return None

        # Display results
        self.display_compact_results(all_predictions)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')

        # Flatten predictions for CSV
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
                    'confidence_level': pred['final_insights']['confidence_level'],
                    'form_analysis': pred['final_insights']['form_analysis'],
                    'team_chemistry': pred['final_insights']['team_chemistry'],
                    'matchup_history': pred['final_insights']['matchup_history']
                })

        # Save to predictions directory (not cleaned data)
        output_file = f"../data/predictions/final_predictions_{timestamp}.csv"
        pd.DataFrame(flattened_predictions).to_csv(output_file, index=False)
        print(f"\n💾 Saved to: {output_file}")
        print(f"📊 Total predictions: {len(flattened_predictions)}")
        print(f"📁 Predictions saved to dedicated predictions folder (not cleaned data)")

        return all_predictions

    def display_compact_results(self, predictions):
        """Display compact betting recommendations."""
        if not predictions:
            print("\n🤷 No high-confidence predictions")
            return

        print(f"\n🏆 NBA BETTING RECOMMENDATIONS")
        print("=" * 80)
        print(f"📊 Total players: {len(predictions)}")

        over_count = sum(1 for p in predictions if
            any(d['recommendation'] == 'OVER' for d in p['predictions'].values()))
        under_count = sum(1 for p in predictions if
            any(d['recommendation'] == 'UNDER' for d in p['predictions'].values()))

        print(f"⏱️  Consensus: {'OVER' if over_count > under_count else 'UNDER'} ({over_count} OVER, {under_count} UNDER)")

        for pred in predictions:
            insights = pred['final_insights']

            print(f"\n🏀 {pred['player_name'].upper()}")

            # Show all markets for this player
            for market, data in pred['predictions'].items():
                print(f"   {market.upper():8} | Line: {data['prop_line']:5} | Pred: {data['predicted_value']:5.1} ({data['line_diff']:+5.1}) | {data['recommendation']:5}")

            print(f"   📊 Confidence: {pred['predictions']['points']['confidence']:.1%} ({insights['confidence_level']})")
            print(f"   💰 Best Odds: {pred['predictions']['points']['bookmaker']} ({pred['predictions']['points']['over_odds']:+})")

            print(f"\n   🔍 ANALYSIS:")
            print(f"      • Form: {insights['form_analysis']}")
            print(f"      • Chemistry: {insights['team_chemistry']}")
            print(f"      • Matchup: {insights['matchup_history']}")

        print("\n" + "=" * 80)

def main():
    """Run the fixed predictions system."""
    if not os.getenv('ODDS_API_KEY'):
        print("❌ Missing API key!")
        print("\nSet ODDS_API_KEY environment variable")
        return

    try:
        system = FixedNBAPredictionsSystem()
        predictions = system.run_final_predictions()

        if predictions:
            print(f"\n🎉 System complete! Generated predictions for {len(predictions)} players")
        else:
            print("\n📊 Complete. No predictions available today.")

    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()