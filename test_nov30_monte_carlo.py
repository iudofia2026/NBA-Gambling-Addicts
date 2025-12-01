#!/usr/bin/env python3
"""
Monte Carlo November 30, 2025 Test
Compare Monte Carlo predictions vs original predictions against real betting lines
"""

import pandas as pd
import numpy as np
from datetime import datetime

class MonteCarloNov30Test:
    """Monte Carlo simulation for November 30, 2025 predictions"""

    def __init__(self):
        # Same betting lines and actual results as original test
        self.betting_lines = {
            'Karl-Anthony Towns': {
                'points': {'line': 23.5, 'over_odds': -122, 'under_odds': -110},
                'rebounds': {'line': 11.5, 'over_odds': 103, 'under_odds': -135},
                'assists': {'line': 3.5, 'over_odds': 126, 'under_odds': -155}
            },
            'Rudy Gobert': {
                'points': {'line': 13.5, 'over_odds': -115, 'under_odds': -105},
                'rebounds': {'line': 14.5, 'over_odds': -110, 'under_odds': -120},
                'assists': {'line': 1.5, 'over_odds': 110, 'under_odds': -140}
            },
            'Mikal Bridges': {
                'points': {'line': 19.5, 'over_odds': -105, 'under_odds': -125},
                'rebounds': {'line': 4.5, 'over_odds': -115, 'under_odds': -115},
                'assists': {'line': 2.5, 'over_odds': 100, 'under_odds': -130}
            },
            'Kevin Durant': {
                'points': {'line': 27.5, 'over_odds': -110, 'under_odds': -120},
                'rebounds': {'line': 7.5, 'over_odds': -105, 'under_odds': -125},
                'assists': {'line': 5.5, 'over_odds': -115, 'under_odds': -115}
            },
            'James Harden': {
                'points': {'line': 21.5, 'over_odds': -108, 'under_odds': -122},
                'rebounds': {'line': 5.5, 'over_odds': -135, 'under_odds': 105},
                'assists': {'line': 8.5, 'over_odds': -120, 'under_odds': -110}
            }
        }

        self.actual_results = {
            'Karl-Anthony Towns': {'points': 25, 'rebounds': 15, 'assists': 6},
            'Rudy Gobert': {'points': 12, 'rebounds': 18, 'assists': 1},
            'Mikal Bridges': {'points': 18, 'rebounds': 5, 'assists': 3},
            'Kevin Durant': {'points': 24, 'rebounds': 8, 'assists': 4},
            'James Harden': {'points': 19, 'rebounds': 4, 'assists': 7}
        }

        # Original model predictions for comparison
        self.original_predictions = {
            'Karl-Anthony Towns': {'points': 27.8, 'rebounds': 14.2, 'assists': 4.1},
            'Rudy Gobert': {'points': 14.5, 'rebounds': 16.8, 'assists': 1.3},
            'Mikal Bridges': {'points': 20.1, 'rebounds': 5.3, 'assists': 2.8},
            'Kevin Durant': {'points': 28.9, 'rebounds': 8.1, 'assists': 6.2},
            'James Harden': {'points': 23.2, 'rebounds': 5.8, 'assists': 9.1}
        }

        # Historical baseline data for variance calculations
        self.historical_baseline = {
            'Karl-Anthony Towns': {
                'points': {'mean': 24.5, 'std': 6.2, 'variance': 0.25},
                'rebounds': {'mean': 12.8, 'std': 4.1, 'variance': 0.32},
                'assists': {'mean': 4.2, 'std': 2.1, 'variance': 0.50}
            },
            'Rudy Gobert': {
                'points': {'mean': 13.2, 'std': 3.8, 'variance': 0.29},
                'rebounds': {'mean': 15.5, 'std': 3.2, 'variance': 0.21},
                'assists': {'mean': 1.3, 'std': 0.8, 'variance': 0.62}
            },
            'Mikal Bridges': {
                'points': {'mean': 19.8, 'std': 5.1, 'variance': 0.26},
                'rebounds': {'mean': 5.2, 'std': 2.3, 'variance': 0.44},
                'assists': {'mean': 2.9, 'std': 1.4, 'variance': 0.48}
            },
            'Kevin Durant': {
                'points': {'mean': 27.5, 'std': 6.8, 'variance': 0.25},
                'rebounds': {'mean': 7.8, 'std': 2.9, 'variance': 0.37},
                'assists': {'mean': 5.6, 'std': 2.2, 'variance': 0.39}
            },
            'James Harden': {
                'points': {'mean': 21.8, 'std': 5.9, 'variance': 0.27},
                'rebounds': {'mean': 5.8, 'std': 2.1, 'variance': 0.36},
                'assists': {'mean': 8.7, 'std': 3.1, 'variance': 0.36}
            }
        }

    def run_monte_carlo_simulation(self, player_name, stat_type, baseline, num_simulations=100):
        """
        Run 100 Monte Carlo simulations for a player's stat
        """

        # Get player's historical variance
        variance = self.historical_baseline.get(player_name, {}).get(stat_type, {}).get('variance', 0.3)

        # Elite players have lower variance (more consistent)
        elite_players = ['Kevin Durant', 'James Harden', 'Nikola Jokic', 'LeBron James', 'Stephen Curry']
        if player_name in elite_players:
            variance *= 0.8  # Reduce variance by 20% for elite players

        # Run Monte Carlo simulations
        np.random.seed(42)  # For reproducible results
        simulations = np.random.normal(
            loc=baseline,
            scale=baseline * variance,
            size=num_simulations
        )

        # Ensure no negative values
        simulations = np.maximum(simulations, 0)

        # Calculate statistics
        mean_prediction = np.mean(simulations)
        median_prediction = np.median(simulations)
        std_prediction = np.std(simulations)

        # Get prop line
        prop_line = self.betting_lines[player_name][stat_type]['line']

        # Calculate probabilities
        prob_over = np.mean(simulations > prop_line)
        prob_under = np.mean(simulations < prop_line)
        prob_push = np.mean(np.abs(simulations - prop_line) < 0.5)

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

        # Calculate confidence intervals
        confidence_95 = np.percentile(simulations, [2.5, 97.5])

        return {
            'player_name': player_name,
            'stat_type': stat_type,
            'baseline': baseline,
            'mean_prediction': mean_prediction,
            'median_prediction': median_prediction,
            'std_prediction': std_prediction,
            'prop_line': prop_line,
            'recommendation': recommendation,
            'confidence': confidence,
            'prob_over': prob_over,
            'prob_under': prob_under,
            'prob_push': prob_push,
            'confidence_95_low': confidence_95[0],
            'confidence_95_high': confidence_95[1],
            'variance': variance,
            'simulations': simulations
        }

    def test_nov30_monte_carlo(self):
        """Run Monte Carlo test for November 30th predictions"""

        print("🎲 November 30, 2025 - Monte Carlo NBA Prediction Test")
        print("=" * 70)
        print("Testing Monte Carlo predictions (100 simulations) vs original predictions")
        print()

        # Track results
        mc_results = []
        original_results = []
        mc_predictions = 0
        mc_correct_count = 0
        original_predictions = 0
        original_correct_count = 0
        mc_profit = 0
        original_profit = 0

        # Test each player and stat
        for player_name, stats in self.betting_lines.items():
            print(f"🎯 {player_name}")
            print("   Stat      | Line  | Actual | Original | MC Mean | MC Rec | Orig | MC")

            for stat_type, line_data in stats.items():
                # Get actual result
                actual = self.actual_results[player_name][stat_type]
                prop_line = line_data['line']

                # Original prediction test
                orig_pred = self.original_predictions[player_name][stat_type]
                orig_result = "OVER" if actual > prop_line else "UNDER" if actual < prop_line else "PUSH"
                orig_prediction = "OVER" if orig_pred > prop_line else "UNDER" if orig_pred < prop_line else "PUSH"
                orig_correct = (orig_result == orig_prediction and orig_result != "PUSH")

                # Monte Carlo simulation
                mc_result = self.run_monte_carlo_simulation(
                    player_name, stat_type, self.original_predictions[player_name][stat_type]
                )

                # Monte Carlo test
                mc_correct = (orig_result == mc_result['recommendation'] and orig_result != "PUSH")

                # Update counters
                if mc_result['recommendation'] != "PASS":
                    mc_predictions += 1
                    if mc_correct:
                        mc_correct_count += 1

                if orig_prediction != "PUSH":
                    original_predictions += 1
                    if orig_correct:
                        original_correct_count += 1

                # Calculate profit (simplified)
                mc_profit += 100 if mc_correct and mc_result['recommendation'] != "PASS" else -100
                original_profit += 100 if orig_correct else -100

                # Store results
                mc_results.append({
                    'Player': player_name,
                    'Stat': stat_type,
                    'Line': prop_line,
                    'Actual': actual,
                    'Original_Prediction': orig_pred,
                    'Original_Result': orig_prediction,
                    'Original_Correct': orig_correct,
                    'MC_Mean': round(mc_result['mean_prediction'], 1),
                    'MC_Recommendation': mc_result['recommendation'],
                    'MC_Correct': mc_correct,
                    'MC_Confidence': round(mc_result['confidence'], 3),
                    'MC_Prob_Over': round(mc_result['prob_over'], 3),
                    'MC_CI_Low': round(mc_result['confidence_95_low'], 1),
                    'MC_CI_High': round(mc_result['confidence_95_high'], 1),
                    'MC_Std': round(mc_result['std_prediction'], 2)
                })

                # Display results
                orig_status = "✅" if orig_correct else "❌"
                mc_status = "✅" if mc_correct else "❌"

                print(f"   {stat_type.title():<8} | {prop_line:<5} | {actual:<6} | {orig_pred:<8.1f} | {mc_result['mean_prediction']:<7.1f} | {mc_result['recommendation']:<6} | {orig_status:<4} | {mc_status:<2}")

                # Show Monte Carlo details
                if mc_result['recommendation'] == "PASS":
                    print(f"            PASS: Over={mc_result['prob_over']:.1%}, Under={mc_result['prob_under']:.1%}")
                else:
                    print(f"            Confidence: {mc_result['confidence']:.1%}, Range: {mc_result['confidence_95_low']:.1f}-{mc_result['confidence_95_high']:.1f}")

            print()

        # Calculate final metrics
        mc_accuracy = (mc_correct_count / mc_predictions) * 100 if mc_predictions > 0 else 0
        original_accuracy = (original_correct_count / original_predictions) * 100 if original_predictions > 0 else 0
        mc_roi = (mc_profit / (mc_predictions * 100)) * 100 if mc_predictions > 0 else 0
        original_roi = (original_profit / (original_predictions * 100)) * 100 if original_predictions > 0 else 0

        # Display comparison summary
        print("📈 PERFORMANCE COMPARISON")
        print("=" * 50)
        print("ORIGINAL MODEL:")
        print(f"  Predictions: {original_predictions}")
        print(f"  Correct: {original_correct_count}")
        print(f"  Accuracy: {original_accuracy:.1f}%")
        print(f"  Profit: ${original_profit:.0f}")
        print(f"  ROI: {original_roi:.1f}%")
        print()
        print("MONTE CARLO MODEL:")
        print(f"  Predictions: {mc_predictions} (PASS recommendations excluded)")
        print(f"  Correct: {mc_correct_count}")
        print(f"  Accuracy: {mc_accuracy:.1f}%")
        print(f"  Profit: ${mc_profit:.0f}")
        print(f"  ROI: {mc_roi:.1f}%")
        print()

        # Determine winner
        if mc_accuracy > original_accuracy:
            improvement = mc_accuracy - original_accuracy
            print(f"🏆 MONTE CARLO WINS by {improvement:.1f}% accuracy!")
        elif original_accuracy > mc_accuracy:
            difference = original_accuracy - mc_accuracy
            print(f"📊 ORIGINAL WINS by {difference:.1f}% accuracy")
        else:
            print("🤝 TIE - Both models performed equally")

        # Save detailed results
        df_mc_results = pd.DataFrame(mc_results)
        output_file = f"data/predictions_archive/monte_carlo_vs_original_nov30_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_mc_results.to_csv(output_file, index=False)

        print(f"\n📁 Detailed comparison saved to: {output_file}")

        return {
            'original_accuracy': original_accuracy,
            'mc_accuracy': mc_accuracy,
            'original_profit': original_profit,
            'mc_profit': mc_profit,
            'original_predictions': original_predictions,
            'mc_predictions': mc_predictions
        }

def main():
    """Run the comparison test"""
    print("Comparing Original vs Monte Carlo predictions for November 30, 2025")
    print()

    # Load the original test results for comparison
    try:
        original_test_file = "data/predictions_archive/nov30_accuracy_test_20251201_184133.csv"
        if os.path.exists(original_test_file):
            print(f"📊 Original test results loaded from: {original_test_file}")
            original_df = pd.read_csv(original_test_file)
            print(f"   Original accuracy: {original_df['Correct'].mean() * 100:.1f}%")
            print()
    except:
        pass

    # Run Monte Carlo comparison
    test = MonteCarloNov30Test()
    results = test.test_nov30_monte_carlo()

    return results

if __name__ == "__main__":
    import os  # Import for file checking
    main()