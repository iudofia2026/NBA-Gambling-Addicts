"""
ADVANCED NBA PREDICTION SYSTEM WITH MONTE CARLO SIMULATION
Uses 100 predictions per player to determine statistical confidence
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the main predictor
from odds_api_client import NBAOddsClient

class MonteCarloNBAPredictor:
    """
    Enhanced predictor using Monte Carlo simulation for more accurate predictions.
    Runs 100 simulations per player to calculate statistical distributions.
    """

    def __init__(self, api_key=None):
        """Initialize the Monte Carlo predictor."""
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError("API key required")

        # Initialize components
        self.odds_client = NBAOddsClient(self.api_key)
        self.historical_data = None

        print("=" * 60)
        print("🎲 MONTE CARLO NBA PREDICTION SYSTEM")
        print("   ✓ 100 simulations per player")
        print("   ✓ Statistical confidence intervals")
        print("   ✓ Variance and distribution analysis")
        print("   ✓ Probability-based recommendations")
        print("=" * 60)

    def load_historical_data(self):
        """Load historical data for variance calculations."""
        data_file = 'data/processed/engineered_features.csv'
        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            return False

        self.historical_data = pd.read_csv(data_file)
        self.historical_data['gameDate'] = pd.to_datetime(self.historical_data['gameDate'], errors='coerce')
        print(f"✅ Loaded historical data: {len(self.historical_data):,} games")
        return True

    def calculate_player_variance(self, player_name, stat_type, num_simulations=100):
        """
        Calculate variance for a player's performance based on historical data.
        This determines how much variation to add to each simulation.
        """

        # Map stat types
        stat_column = {
            'points': 'points',
            'rebounds': 'reboundsTotal',
            'assists': 'assists'
        }.get(stat_type)

        if not stat_column:
            return 0.1  # Default variance

        # Get player's historical data
        player_data = self.historical_data[self.historical_data['fullName'] == player_name]
        if player_data.empty:
            return 0.15  # Higher variance for unknown players

        # Use only recent starter games (last 20)
        starter_games = player_data[player_data['numMinutes'] >= 28.0].tail(20)

        if len(starter_games) < 5:
            starter_games = player_data.tail(10)  # Fallback to recent games

        if len(starter_games) < 3:
            return 0.2  # High variance for limited data

        # Calculate statistical variance
        values = starter_games[stat_column].values
        variance = np.var(values) if len(values) > 1 else 1.0
        std_dev = np.std(values) if len(values) > 1 else 1.0

        # Normalize variance as percentage of mean
        mean_value = np.mean(values)
        normalized_variance = (std_dev / mean_value) if mean_value > 0 else 0.2

        # Cap variance at reasonable levels
        normalized_variance = min(max(normalized_variance, 0.05), 0.4)

        print(f"      📊 {player_name} {stat_type}: variance={normalized_variance:.3f} (std={std_dev:.1f}, mean={mean_value:.1f})")

        return normalized_variance

    def run_monte_carlo_simulation(self, player_name, stat_type, prop_line, num_simulations=100):
        """
        Run 100 Monte Carlo simulations to get a distribution of possible outcomes.
        """

        # Get player's robust baseline (same method as original)
        prediction = self.get_robust_prediction(player_name, stat_type, prop_line)
        if not prediction:
            return None

        baseline = prediction['baseline']
        variance = self.calculate_player_variance(player_name, stat_type, num_simulations)

        # Run Monte Carlo simulations
        np.random.seed(42)  # For reproducible results

        # Generate random variations based on player's historical variance
        # Using normal distribution with player-specific variance
        simulations = np.random.normal(
            loc=baseline,
            scale=baseline * variance,
            size=num_simulations
        )

        # Ensure no negative values
        simulations = np.maximum(simulations, 0)

        # Apply some realistic constraints
        # Elite players have floor (minimum performance)
        elite_players = ['Nikola Jokic', 'Kevin Durant', 'James Harden', 'LeBron James',
                        'Stephen Curry', 'Jayson Tatum', 'Devin Booker']

        if player_name in elite_players:
            # Elite players have performance floors
            min_points = {'Nikola Jokic': 15, 'Kevin Durant': 20, 'James Harden': 15,
                          'LeBron James': 18, 'Stephen Curry': 18, 'Jayson Tatum': 18,
                          'Devin Booker': 16}.get(player_name, 10)

            if stat_type == 'points':
                simulations = np.maximum(simulations, min_points)
            elif stat_type == 'rebounds':
                simulations = np.maximum(simulations, 5)
            elif stat_type == 'assists':
                simulations = np.maximum(simulations, 3)

        # Calculate statistics from simulations
        mean_prediction = np.mean(simulations)
        median_prediction = np.median(simulations)
        std_prediction = np.std(simulations)

        # Calculate probability of going over/under the line
        prob_over = np.mean(simulations > prop_line)
        prob_under = np.mean(simulations < prop_line)
        prob_exact = np.mean(np.abs(simulations - prop_line) < 0.5)  # Within 0.5 points = push

        # Calculate confidence intervals
        confidence_95 = np.percentile(simulations, [2.5, 97.5])
        confidence_90 = np.percentile(simulations, [5, 95])
        confidence_80 = np.percentile(simulations, [10, 90])

        # Determine recommendation based on probability
        if prob_over > 0.55:
            recommendation = "OVER"
            confidence = min(prob_over, 0.95)
        elif prob_under > 0.55:
            recommendation = "UNDER"
            confidence = min(prob_under, 0.95)
        else:
            recommendation = "PASS"  # Too close to call
            confidence = max(prob_over, prob_under)

        return {
            'baseline': baseline,
            'mean_prediction': mean_prediction,
            'median_prediction': median_prediction,
            'std_prediction': std_prediction,
            'prop_line': prop_line,
            'recommendation': recommendation,
            'confidence': confidence,
            'prob_over': prob_over,
            'prob_under': prob_under,
            'prob_push': prob_exact,
            'confidence_intervals': {
                '95%': confidence_95,
                '90%': confidence_90,
                '80%': confidence_80
            },
            'simulations': simulations,
            'variance': variance,
            # Add original prediction details
            'recent_avg': prediction.get('recent_avg', 0),
            'season_avg': prediction.get('season_avg', 0)
        }

    def get_robust_prediction(self, player_name, stat_type, prop_line):
        """
        Get robust baseline prediction (copied from original).
        This provides the mean for our Monte Carlo simulations.
        """

        # Get player data
        player_data = self.historical_data[self.historical_data['fullName'] == player_name]
        if player_data.empty:
            return None

        # Get stat column name
        stat_column = {
            'points': 'points',
            'rebounds': 'reboundsTotal',
            'assists': 'assists'
        }.get(stat_type)

        if not stat_column or stat_column not in player_data.columns:
            return None

        # AGGRESSIVE FILTERING: Only use STARTER-LEVEL games (28+ minutes)
        starter_games = player_data[player_data['numMinutes'] >= 28.0]

        if len(starter_games) < 5:
            return None

        # Use only recent starter-level performances
        recent_starters = starter_games.tail(10)  # Last 10 starter games
        season_starters = starter_games.tail(20)  # Last 20 starter games

        if len(recent_starters) < 3:
            return None

        # Calculate averages from STARTER games only
        recent_avg = recent_starters[stat_column].mean()
        season_avg = season_starters[stat_column].mean()

        # Baseline prediction: 70% recent, 30% season
        baseline = (recent_avg * 0.7) + (season_avg * 0.3)

        # ELITE PLAYER BOOST
        elite_players = {
            'Nikola Jokic': {'points': 26, 'rebounds': 14, 'assists': 9},
            'Kevin Durant': {'points': 30, 'rebounds': 6, 'assists': 4},
            'James Harden': {'points': 22, 'rebounds': 6, 'assists': 8},
            'Devin Booker': {'points': 24, 'rebounds': 4, 'assists': 7},
            'LeBron James': {'points': 25, 'rebounds': 7, 'assists': 8},
            'Stephen Curry': {'points': 27, 'rebounds': 4, 'assists': 6},
            'Jayson Tatum': {'points': 27, 'rebounds': 7, 'assists': 4}
        }

        if player_name in elite_players:
            elite_baseline = elite_players[player_name].get(stat_type, baseline)
            if baseline < elite_baseline * 0.75:
                baseline = elite_baseline * 0.9

        return {
            'baseline': baseline,
            'recent_avg': recent_avg,
            'season_avg': season_avg
        }

    def make_monte_carlo_prediction(self, player_name, prop_data, game_context):
        """
        Make prediction using Monte Carlo simulation for all markets.
        """

        predictions = {}

        for market_type, market_props in prop_data.items():
            print(f"      🎲 Running Monte Carlo for {market_type} (100 simulations)...")

            # Convert market type to stat type
            stat_type = None
            if market_type.lower() == 'player_points':
                stat_type = 'points'
            elif market_type.lower() == 'player_rebounds':
                stat_type = 'rebounds'
            elif market_type.lower() == 'player_assists':
                stat_type = 'assists'
            else:
                continue

            # Find best odds
            prop_df = pd.DataFrame(market_props)
            prop_df['over_odds_numeric'] = pd.to_numeric(prop_df['over_odds'], errors='coerce')
            best_prop = prop_df.loc[prop_df['over_odds_numeric'].idxmax()].to_dict()

            # Run Monte Carlo simulation
            mc_result = self.run_monte_carlo_simulation(
                player_name, stat_type, best_prop['prop_line'], num_simulations=100
            )

            if mc_result:
                predictions[market_type] = {
                    'market_type': market_type,
                    'prop_line': best_prop['prop_line'],
                    'recommendation': mc_result['recommendation'],
                    'predicted_value': round(mc_result['mean_prediction'], 1),
                    'median_value': round(mc_result['median_prediction'], 1),
                    'line_diff': round(mc_result['mean_prediction'] - best_prop['prop_line'], 1),
                    'confidence': round(mc_result['confidence'], 3),
                    'over_odds': best_prop['over_odds'],
                    'bookmaker': best_prop['bookmaker'],
                    'game_time': best_prop['game_time'],
                    # Monte Carlo specific data
                    'prob_over': round(mc_result['prob_over'], 3),
                    'prob_under': round(mc_result['prob_under'], 3),
                    'prob_push': round(mc_result['prob_push'], 3),
                    'std_deviation': round(mc_result['std_prediction'], 2),
                    'variance': round(mc_result['variance'], 3),
                    'confidence_95_low': round(mc_result['confidence_intervals']['95%'][0], 1),
                    'confidence_95_high': round(mc_result['confidence_intervals']['95%'][1], 1),
                    'baseline': round(mc_result['baseline'], 1),
                    'recent_avg': round(mc_result['recent_avg'], 1),
                    'season_avg': round(mc_result['season_avg'], 1)
                }

                print(f"      ✅ {stat_type}: {mc_result['recommendation']} {best_prop['prop_line']} "
                      f"(mean: {mc_result['mean_prediction']:.1f}, prob: {mc_result['prob_over']:.1%} over)")
            else:
                print(f"      ❌ Could not run Monte Carlo for {stat_type}")

        if not predictions:
            return None

        print(f"   ✅ Generated Monte Carlo predictions for {player_name}")

        return {
            'player_name': player_name,
            'predictions': predictions,
            'method': 'monte_carlo_simulation',
            'simulations_per_prediction': 100
        }

    def run_predictions(self):
        """Run the Monte Carlo prediction system."""

        # Load data
        if not self.load_historical_data():
            return None

        # Fetch today's props
        print("\n📡 FETCHING TODAY'S NBA PLAYER PROPS")
        props_df = self.odds_client.get_all_todays_player_props()

        if props_df.empty:
            print("❌ No player props available")
            return None

        # Group props by player (same as original)
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

        # Generate Monte Carlo predictions
        print(f"\n🎲 GENERATING MONTE CARLO PREDICTIONS")
        print("-" * 50)

        all_predictions = []

        for player_name, player_markets in player_props.items():
            print(f"\n🎯 {player_name}")

            # Get game context (same as original)
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

            # Flatten markets for prediction (same as original)
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
                prediction = self.make_monte_carlo_prediction(player_name, flattened_markets, game_context)
                if prediction:
                    all_predictions.append(prediction)

        if not all_predictions:
            print("\n❌ No predictions generated")
            return None

        print(f"\n✅ Successfully generated Monte Carlo predictions for {len(all_predictions)} players")

        # Display results
        self.display_monte_carlo_results(all_predictions)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_file = f"data/predictions/monte_carlo_predictions_{timestamp}.csv"

        # Flatten for CSV with Monte Carlo data
        flattened_predictions = []
        for pred in all_predictions:
            if 'predictions' not in pred:
                continue
            for market, data in pred['predictions'].items():
                flattened_predictions.append({
                    'player_name': pred['player_name'],
                    'market_type': market,
                    'prop_line': data['prop_line'],
                    'recommendation': data['recommendation'],
                    'predicted_value': data['predicted_value'],
                    'median_value': data['median_value'],
                    'line_diff': data['line_diff'],
                    'confidence': data['confidence'],
                    'over_odds': data['over_odds'],
                    'bookmaker': data['bookmaker'],
                    'game_time': data['game_time'],
                    # Monte Carlo specific columns
                    'prob_over': data['prob_over'],
                    'prob_under': data['prob_under'],
                    'prob_push': data['prob_push'],
                    'std_deviation': data['std_deviation'],
                    'variance': data['variance'],
                    'confidence_95_low': data['confidence_95_low'],
                    'confidence_95_high': data['confidence_95_high'],
                    'baseline': data['baseline'],
                    'recent_avg': data['recent_avg'],
                    'season_avg': data['season_avg']
                })

        pd.DataFrame(flattened_predictions).to_csv(output_file, index=False)
        print(f"\n💾 Monte Carlo results saved to: {output_file}")
        print(f"📊 Total predictions: {len(flattened_predictions)}")

        return all_predictions

    def display_monte_carlo_results(self, predictions):
        """Display Monte Carlo betting recommendations."""

        if not predictions:
            print("\n🤷 No predictions available")
            return

        print(f"\n🎲 MONTE CARLO NBA BETTING RECOMMENDATIONS")
        print("=" * 100)
        print(f"📊 Total players: {len(predictions)} | 🔬 100 simulations per prediction")

        # Count recommendations
        over_count = sum(1 for p in predictions if
            any(d['recommendation'] == 'OVER' for d in p['predictions'].values()))
        under_count = sum(1 for p in predictions if
            any(d['recommendation'] == 'UNDER' for d in p['predictions'].values()))
        pass_count = sum(1 for p in predictions if
            any(d['recommendation'] == 'PASS' for d in p['predictions'].values()))

        print(f"⏱️  Consensus: {max([('OVER', over_count), ('UNDER', under_count), ('PASS', pass_count)], key=lambda x: x[1])[0]} "
              f"({over_count} OVER, {under_count} UNDER, {pass_count} PASS)")

        for pred in predictions:
            print(f"\n🏀 {pred['player_name'].upper()}")

            # Show predictions with Monte Carlo data
            for market, data in pred['predictions'].items():
                market_name = market.replace('player_', '').upper()
                print(f"   {market_name:8} | Line: {data['prop_line']:5} | "
                      f"Mean: {data['predicted_value']:5.1f} | "
                      f"95% CI: [{data['confidence_95_low']:4.1f}-{data['confidence_95_high']:4.1f}] | "
                      f"P(Over): {data['prob_over']:.1%} | "
                      f"{data['recommendation']:5}")

            # Show best market
            best_market = max(pred['predictions'].values(), key=lambda x: x['confidence'])
            print(f"   📊 Best Confidence: {best_market['confidence']:.1%} | "
                  f"💰 Best Odds: {best_market['bookmaker']} (+{best_market['over_odds']})")

        print("\n" + "=" * 100)

def main():
    """Run the Monte Carlo predictions system."""
    if not os.getenv('ODDS_API_KEY'):
        print("❌ Missing API key!")
        print("\nSet ODDS_API_KEY environment variable")
        return

    try:
        predictor = MonteCarloNBAPredictor()
        predictions = predictor.run_predictions()

        if predictions:
            print(f"\n🎲 Monte Carlo system complete! Generated predictions for {len(predictions)} players")
        else:
            print("\n📊 Complete. No predictions available today.")

    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()