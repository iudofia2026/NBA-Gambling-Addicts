"""
ENHANCED NBA PREDICTIONS SYSTEM
Migrating experimental features from advanced_analytics_v6.py
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

class EnhancedNBAPredictionsSystem:
    """Enhanced predictions system with migrated experimental features."""

    def __init__(self, api_key=None):
        """Initialize the enhanced predictions system."""
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError("API key required")

        self.odds_client = NBAOddsClient(self.api_key)
        self.models = {}
        self.historical_data = None

        print("=" * 60)
        print("🚀 ENHANCED NBA PREDICTIONS SYSTEM")
        print("   Migrated Features:")
        print("   ✓ All original 3 iterations")
        print("   ✓ Seasonal Trends (Iteration 6)")
        print("   ✓ Peak Performance Analysis (Iteration 7)")
        print("   ✓ Enhanced Team Dynamics (Iteration 8)")
        print("   ✓ Situational Pressure/Clutch (Iteration 9)")
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

    # ===== ORIGINAL FEATURES =====
    def analyze_player_form(self, player_name, days_back=15):
        """Analyze player's current form (Original Iteration 3)."""
        player_data = self.historical_data[
            self.historical_data['fullName'] == player_name
        ].tail(days_back).copy()

        if player_data.empty:
            return self._get_default_form()

        # Calculate momentum indicators
        recent_games = player_data.tail(5)
        older_games = player_data.head(max(0, len(player_data) - 5))

        momentum = {}

        # Hot/Cold streak detection
        if len(recent_games) >= 3:
            momentum['current_streak'] = self._detect_streak(recent_games['points'])
            momentum['streak_length'] = len(recent_games)
            momentum['streak_intensity'] = np.std(recent_games['points']) / recent_games['points'].mean() if recent_games['points'].mean() > 0 else 0
        else:
            momentum['current_streak'] = 'neutral'
            momentum['streak_length'] = len(recent_games)
            momentum['streak_intensity'] = 0

        # Recent performance trend
        if len(recent_games) >= 5:
            momentum['trend'] = np.polyfit(range(len(recent_games)), recent_games['points'], 1)[0]
            momentum['trend_strength'] = abs(momentum['trend'])
        else:
            momentum['trend'] = 0
            momentum['trend_strength'] = 0

        # Confidence level based on consistency
        momentum['form_confidence'] = 1 - (player_data['points'].std() / player_data['points'].mean() if player_data['points'].mean() > 0 else 0.5)

        return momentum

    # ===== MIGRATED FEATURES =====

    # Iteration 6: Seasonal Trends
    def calculate_seasonal_trends(self, player_name):
        """Seasonal progression and peak performance patterns."""
        player_data = self.historical_data[self.historical_data['fullName'] == player_name].copy()

        if len(player_data) < 30:
            return self._default_seasonal_features()

        features = {}

        # Season progression (month-by-month)
        player_data['month'] = pd.to_datetime(player_data['gameDate']).dt.month
        monthly_avg = player_data.groupby('month')['points'].mean()

        # Identify peak months
        peak_months = monthly_avg.nlargest(3)
        features['seasonal_peak_months'] = peak_months.index.tolist()
        features['seasonal_peak_avg'] = peak_months.mean()

        # Recent vs early season comparison
        recent_games = player_data.tail(20)
        if len(recent_games) >= 10:
            early_season = player_data.head(20)
            features['season_improvement'] = recent_games['points'].mean() - early_season['points'].mean()
        else:
            features['season_improvement'] = 0

        # Current month performance
        current_month = datetime.now().month
        if current_month in monthly_avg.index:
            features['current_month_avg'] = monthly_avg[current_month]
            features['seasonal_factor'] = features['current_month_avg'] / monthly_avg.mean()
        else:
            features['current_month_avg'] = player_data['points'].mean()
            features['seasonal_factor'] = 1.0

        return features

    # Iteration 7: Peak Performance Analysis
    def calculate_peak_performance_factors(self, player_name):
        """Performance peaks and variance analysis."""
        player_data = self.historical_data[self.historical_data['fullName'] == player_name]

        features = {}

        if len(player_data) > 0:
            # Performance distribution
            points_mean = player_data['points'].mean()
            points_std = player_data['points'].std()

            features['performance_consistency'] = 1 - (points_std / points_mean) if points_mean > 0 else 0
            features['performance_volatility'] = points_std

            # Peak performance metrics
            top_10_percentile = player_data['points'].quantile(0.90)
            features['peak_performance'] = top_10_percentile
            features['peak_to_avg_ratio'] = top_10_percentile / points_mean if points_mean > 0 else 1

            # Recent form factor
            recent_5 = player_data.tail(5)
            if len(recent_5) >= 3:
                features['recent_form'] = recent_5['points'].mean() / points_mean if points_mean > 0 else 1
                features['breakout_potential'] = (top_10_percentile - recent_5['points'].mean()) / points_mean if points_mean > 0 else 0
            else:
                features['recent_form'] = 1
                features['breakout_potential'] = 0

        return features

    # Iteration 8: Enhanced Team Dynamics
    def analyze_enhanced_team_chemistry(self, player_name, player_team, days_back=10):
        """Enhanced team chemistry and lineup synergy."""
        team_games = self.historical_data[
            (self.historical_data['playerteamName'] == player_team) |
            (self.historical_data['opponentteamName'] == player_team)
        ].tail(days_back)

        if team_games.empty:
            return self._get_default_chemistry()

        chemistry = {}

        # Team momentum (more sophisticated)
        recent_team = team_games.tail(5)
        if len(recent_team) >= 5:
            older_team = team_games.head(len(team_games) - 5)
            if len(older_team) > 0:
                chemistry['team_momentum'] = recent_team['points'].mean() - older_team['points'].mean()
                chemistry['momentum_trend'] = np.polyfit(range(len(recent_team)), recent_team['points'].values, 1)[0]
            else:
                chemistry['team_momentum'] = 0
                chemistry['momentum_trend'] = 0

            # Chemistry metrics
            chemistry['momentum_consistency'] = 1 - (recent_team['points'].std() / recent_team['points'].mean() if recent_team['points'].mean() > 0 else 0.5)
            chemistry['team_scoring_efficiency'] = (recent_team['points'] / recent_team['numMinutes']).mean() if 'numMinutes' in recent_team.columns else 0.5
        else:
            chemistry['team_momentum'] = 0
            chemistry['momentum_trend'] = 0
            chemistry['momentum_consistency'] = 0.5
            chemistry['team_scoring_efficiency'] = 0.5

        # Player's role in team
        player_team_games = team_games[team_games['fullName'] == player_name]
        if len(player_team_games) > 0:
            team_avg = recent_team['points'].mean() if len(recent_team) > 0 else team_data['points'].mean()
            chemistry['team_usage_share'] = player_team_games['points'].mean() / team_avg if team_avg > 0 else 0.2
            chemistry['chemistry_impact'] = min(chemistry['team_usage_share'] * chemistry['momentum_consistency'], 1.0)
            chemistry['role_consistency'] = 1 - (player_team_games['points'].std() / player_team_games['points'].mean() if player_team_games['points'].mean() > 0 else 0.5)
        else:
            chemistry['team_usage_share'] = 0.2
            chemistry['chemistry_impact'] = 0.5
            chemistry['role_consistency'] = 0.5

        return chemistry

    # Iteration 9: Situational Pressure/Clutch Analysis
    def calculate_situational_pressure(self, player_name):
        """High-pressure and clutch situation performance."""
        player_data = self.historical_data[self.historical_data['fullName'] == player_name].copy()

        features = {}

        if len(player_data) > 10:
            # High-pressure games (close games defined by margin)
            if 'plusMinusPoints' in player_data.columns:
                player_data['margin'] = abs(player_data['plusMinusPoints'])
                close_games = player_data[player_data['margin'] <= 5]

                if len(close_games) > 0:
                    features['clutch_performance'] = close_games['points'].mean() / player_data['points'].mean() if player_data['points'].mean() > 0 else 1
                    features['clutch_minutes'] = close_games['numMinutes'].mean() / player_data['numMinutes'].mean() if player_data['numMinutes'].mean() > 0 else 1
                    features['clutch_efficiency'] = (close_games['points'] / close_games['numMinutes']).mean() if len(close_games) > 0 else player_data['points'].mean() / player_data['numMinutes'].mean()
                else:
                    features['clutch_performance'] = 1
                    features['clutch_minutes'] = 1
                    features['clutch_efficiency'] = player_data['points'].mean() / player_data['numMinutes'].mean() if player_data['numMinutes'].mean() > 0 else 0.5

                # Blowout games performance
                blowout_games = player_data[player_data['margin'] >= 15]
                if len(blowout_games) > 0:
                    features['blowout_efficiency'] = blowout_games['points'].mean() / blowout_games['numMinutes'].mean() if blowout_games['numMinutes'].mean() > 0 else 1
                    features['blowout_vs_normal'] = features['blowout_efficiency'] / features['clutch_efficiency'] if features['clutch_efficiency'] > 0 else 1
                else:
                    features['blowout_efficiency'] = 1
                    features['blowout_vs_normal'] = 1
            else:
                # Default values if no margin data
                features['clutch_performance'] = 1
                features['clutch_minutes'] = 1
                features['clutch_efficiency'] = player_data['points'].mean() / player_data['numMinutes'].mean() if player_data['numMinutes'].mean() > 0 else 0.5
                features['blowout_efficiency'] = 1
                features['blowout_vs_normal'] = 1

        return features

    def calculate_enhanced_prediction(self, player_name, game_context, baseline_points):
        """Calculate prediction with all features (original + migrated)."""

        # Get all feature sets
        form = self.analyze_player_form(player_name)
        seasonal = self.calculate_seasonal_trends(player_name)
        peaks = self.calculate_peak_performance_factors(player_name)
        chemistry = self.analyze_enhanced_team_chemistry(
            player_name,
            game_context.get('player_team', '')
        )
        matchup = self.analyze_individual_matchup(player_name, game_context['opponent_team'])
        pressure = self.calculate_situational_pressure(player_name)

        # Enhanced weights (optimized for accuracy)
        weights = {
            'form': 0.20,           # Recent form
            'seasonal': 0.15,       # Seasonal trends
            'peaks': 0.15,          # Peak performance
            'chemistry': 0.15,      # Team chemistry
            'matchup': 0.20,        # Individual matchup
            'pressure': 0.10,       # Situational pressure
            'baseline': 0.05        # Historical baseline
        }

        # Calculate adjustments
        adjustments = {
            'form': form['trend'] * form['form_confidence'],
            'seasonal': (seasonal.get('seasonal_factor', 1) - 1) * baseline_points * seasonal.get('season_improvement', 0),
            'peaks': (peaks.get('recent_form', 1) - 1) * baseline_points + peaks.get('breakout_potential', 0) * 2,
            'chemistry': chemistry['team_momentum'] * chemistry['chemistry_impact'] + chemistry['momentum_trend'] * 2,
            'matchup': (matchup['avg_points_vs_opp'] - baseline_points) * matchup['sample_confidence'],
            'pressure': (pressure.get('clutch_performance', 1) - 1) * baseline_points * pressure.get('clutch_minutes', 1),
            'baseline': 0
        }

        # Calculate predicted points
        predicted_points = baseline_points + sum(adjustments[key] * weights[key] for key in weights)

        # Enhanced confidence calculation
        confidence_factors = {
            'form_conf': form['form_confidence'] * 0.20,
            'seasonal_conf': min(len(seasonal.get('seasonal_peak_months', [])) / 3, 1) * 0.10,
            'peak_conf': peaks.get('performance_consistency', 0.5) * 0.15,
            'chemistry_conf': chemistry['chemistry_impact'] * 0.15,
            'matchup_conf': matchup['sample_confidence'] * 0.20,
            'pressure_conf': pressure.get('clutch_performance', 0.5) * 0.10,
            'base_conf': 0.10
        }

        confidence_score = min(sum(confidence_factors.values()), 1.0)

        return {
            'predicted_points': predicted_points,
            'confidence_score': confidence_score,
            'feature_breakdown': {
                'form_factor': form,
                'seasonal_factor': seasonal,
                'peak_factor': peaks,
                'chemistry_factor': chemistry,
                'matchup_factor': matchup,
                'pressure_factor': pressure
            },
            'adjustments': adjustments,
            'weights': weights
        }

    # Use original analyze_individual_matchup method
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

        # Historical performance vs opponent
        matchup['avg_points_vs_opp'] = matchup_data['points'].mean()
        matchup['over_rate_vs_opp'] = (matchup_data['over_threshold'] == 1).mean()
        matchup['efficiency_vs_opp'] = (matchup_data['points'] / matchup_data['numMinutes']).mean()

        # Recent form in this matchup
        recent_matchups = matchup_data.tail(3)
        if len(recent_matchups) >= 2:
            matchup['recent_trend_vs_opp'] = recent_matchups.tail(1)['points'].iloc[0] - recent_matchups.head(1)['points'].iloc[0]
            matchup['consistency_vs_opp'] = 1 - (recent_matchups['points'].std() / recent_matchups['points'].mean())
        else:
            matchup['recent_trend_vs_opp'] = 0
            matchup['consistency_vs_opp'] = 0.5

        # Sample size confidence
        matchup['sample_confidence'] = min(len(matchup_data) / 10, 1.0)

        return matchup

    # Default methods
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
            'momentum_trend': 0,
            'momentum_consistency': 0.5,
            'team_scoring_efficiency': 0.5,
            'team_usage_share': 0.2,
            'chemistry_impact': 0.5,
            'role_consistency': 0.5
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

    def _default_seasonal_features(self):
        return {
            'seasonal_peak_months': [1, 2, 3],
            'seasonal_peak_avg': 15,
            'season_improvement': 0,
            'current_month_avg': 15,
            'seasonal_factor': 1.0
        }

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

    def run_enhanced_predictions(self):
        """Run the enhanced predictions system."""

        # Load models
        if not self.load_models():
            return None

        # Fetch today's props
        print("\n📡 FETCHING TODAY'S NBA PLAYER PROPS")
        props_df = self.odds_client.get_all_todays_player_props()

        if props_df.empty:
            print("❌ No player props available")
            return None

        # Format props
        formatted_props = self.odds_client.format_for_ml_pipeline(props_df)
        print(f"✅ Found {len(formatted_props)} player prop lines")

        # Generate predictions
        print(f"\n🔮 GENERATING ENHANCED PREDICTIONS")
        predictions = []

        for _, prop in formatted_props.iterrows():
            print(f"\n🎯 Analyzing: {prop['fullName']} - {prop['market_type']}")

            # Determine opponent team
            opponent_team = prop['away_team'] if prop['home_team'] == prop.get('playerteamName', '') else prop['home_team']

            game_context = {
                'game_date': prop['gameDate'],
                'home_team': prop['home_team'],
                'away_team': prop['away_team'],
                'player_team': prop.get('playerteamName', ''),
                'opponent_team': opponent_team
            }

            # Get baseline from historical data
            player_data = self.historical_data[self.historical_data['fullName'] == prop['fullName']]
            if player_data.empty:
                continue

            baseline_points = player_data['points'].tail(10).mean()

            # Calculate enhanced prediction
            result = self.calculate_enhanced_prediction(
                prop['fullName'], game_context, baseline_points
            )

            # Determine recommendation
            if prop['market_type'].lower() == 'points':
                predicted_value = result['predicted_points']
            else:  # For rebounds, use a simplified conversion
                predicted_value = baseline_points * 0.4  # Rough estimate: rebounds ≈ 40% of points

            recommendation = "OVER" if predicted_value > prop['prop_line'] else "UNDER"
            confidence = result['confidence_score']

            # High confidence predictions
            if confidence < 0.65:
                continue

            prediction = {
                'player_name': prop['fullName'],
                'market_type': prop['market_type'],
                'prop_line': prop['prop_line'],
                'recommendation': recommendation,
                'confidence': min(confidence, 0.99),
                'over_odds': prop['over_odds'],
                'bookmaker': prop['bookmaker'],
                'game_time': prop['game_time'],
                'enhanced_insights': {
                    'predicted_value': round(predicted_value, 1),
                    'line_diff': round(predicted_value - prop['prop_line'], 1),
                    'confidence_level': self._get_confidence_level(confidence),
                    'form_analysis': f"{result['feature_breakdown']['form_factor']['current_streak']} streak",
                    'seasonal_factor': f"{result['feature_breakdown']['seasonal_factor'].get('seasonal_factor', 1):.2f}",
                    'peak_performance': f"{result['feature_breakdown']['peak_factor'].get('recent_form', 1):.2f}",
                    'team_chemistry': f"{result['feature_breakdown']['chemistry_factor']['chemistry_impact']:.1%}",
                    'matchup_history': f"{result['feature_breakdown']['matchup_factor']['avg_points_vs_opp']:.1f} avg",
                    'clutch_factor': f"{result['feature_breakdown']['pressure_factor'].get('clutch_performance', 1):.2f}"
                }
            }

            predictions.append(prediction)

        if not predictions:
            print("\n❌ No high-confidence predictions generated")
            return None

        # Display results
        self.display_enhanced_results(predictions)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_file = f"../data/processed/enhanced_predictions_{timestamp}.csv"
        pd.DataFrame(predictions).to_csv(output_file, index=False)
        print(f"\n💾 Saved to: {output_file}")

        return predictions

    def display_enhanced_results(self, predictions):
        """Display enhanced betting recommendations."""
        if not predictions:
            print("\n🤷 No high-confidence predictions")
            return

        print(f"\n🏆 ENHANCED NBA BETTING RECOMMENDATIONS")
        print("=" * 80)
        print(f"📊 Total predictions: {len(predictions)}")
        print(f"⏱️  Model consensus: {'OVER' if sum(1 for p in predictions if p['recommendation'] == 'OVER') > len(predictions)/2 else 'UNDER'}")

        for pred in predictions:
            insights = pred['enhanced_insights']

            print(f"\n🏀 {pred['player_name'].upper()}")
            print(f"   Market: {pred['market_type'].upper()} | Line: {pred['prop_line']}")
            print(f"   Predicted: {insights['predicted_value']} ({insights['line_diff']:+})")
            print(f"   🎯 RECOMMENDATION: {pred['recommendation']}")
            print(f"   📊 Confidence: {pred['confidence']:.1%} ({insights['confidence_level']})")
            print(f"   💰 Best Odds: {pred['bookmaker']} {pred['over_odds']:+}")
            print(f"\n   🔍 ENHANCED ANALYSIS:")
            print(f"      • Form: {insights['form_analysis']}")
            print(f"      • Seasonal Factor: {insights['seasonal_factor']}")
            print(f"      • Peak Performance: {insights['peak_performance']}")
            print(f"      • Team Chemistry: {insights['team_chemistry']}")
            print(f"      • Matchup: {insights['matchup_history']}")
            print(f"      • Clutch Factor: {insights['clutch_factor']}")

        print("\n" + "=" * 80)

    def _get_confidence_level(self, confidence):
        """Convert confidence score to descriptive level."""
        if confidence >= 0.95:
            return "Very High"
        elif confidence >= 0.85:
            return "High"
        elif confidence >= 0.75:
            return "Medium-High"
        else:
            return "Medium"

def main():
    """Run the enhanced predictions system."""
    if not os.getenv('ODDS_API_KEY'):
        print("❌ Missing API key!")
        print("\nSet ODDS_API_KEY environment variable")
        return

    try:
        system = EnhancedNBAPredictionsSystem()
        predictions = system.run_enhanced_predictions()

        if predictions:
            print(f"\n🎉 Enhanced system complete! Generated {len(predictions)} recommendations.")
        else:
            print("\n📊 Complete. No high-confidence predictions today.")

    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()