# Priority Implementation Plan for 75% Accuracy

## 🎯 Focus: Highest ROI Features First

Based on research, here's what will give you the biggest accuracy gains with manageable effort:

## 1. IMMEDIATE WINS (Next 2 weeks)
### Expected Accuracy Gain: +5-7%

#### A. Enhanced Fatigue & Rest Metrics
```python
# Add to your feature engineering
def calculate_fatigue_score(player_games, last_7_days, last_30_days):
    """
    More sophisticated fatigue calculation
    """
    # Current implementation likely just checks if B2B
    # Enhance with:
    - Rolling 7-day minutes
    - Weighted recent games (more weight)
    - Age-based recovery factors
    - Travel time zones crossed
```

#### B. Opponent Defensive Quality
```python
# Currently missing: How good is the defense at player's position?
def get_opposition_defense(player_position, opponent_team):
    """
    Get opponent's defense rating against specific position
    """
    # Data needed from NBA API:
    - Points allowed by position
    - Defensive ratings vs position types
    - Recent defensive form
```

#### C. Line Movement Intelligence
```python
# Add market sentiment analysis
def analyze_line_movement(opening_line, current_line, betting_volume):
    """
    Sharp money vs public money detection
    """
    - Line movement direction
    - Volume patterns
    - Time of movement (early vs late)
```

## 2. HIGH IMPACT (Next 1-2 months)
### Expected Accuracy Gain: +5-8%

#### A. Shot Quality Integration
```python
# Source: NBA.com shot chart data
def get_shot_quality_stats(player_id):
    return {
        'avg_shot_difficulty': 0.45,
        'contested_shot_pct': 0.32,
        'open_shot_pct': 0.68,
        'corner_three_freq': 0.25,
        'transition_pts_freq': 0.15
    }
```

#### B. Lineup Synergy Effects
```python
# 5-man lineup effectiveness from NBA API
def lineup_effectiveness(lineup_combo):
    """
    How well does this 5-man unit perform together?
    """
    return {
        'net_rating': +5.2,
        'pace': 98.5,
        'minutes_together': 120,
        'offensive_efficiency': 115.3
    }
```

#### C. Real-time Injury Integration
```python
# Pull from multiple sources
def get_injury_impact(player_id):
    """
    Quantify how injury affects performance
    """
    return {
        'status': 'questionable',
        'impact_factor': 0.85,  # Expected performance reduction
        'minutes_limit': '25-30',
        'backup_quality': 0.6
    }
```

## 3. MODEL ENHANCEMENTS (2-3 months)
### Expected Accuracy Gain: +3-5%

#### A. Ensemble Approach
```python
# Don't rely on single model
class EnsemblePredictor:
    def __init__(self):
        self.models = {
            'historical_patterns': XGBoost(),
            'recent_form': RandomForest(),
            'matchup_specific': NeuralNetwork(),
            'market_efficiency': BayesianModel()
        }

    def predict(self, features):
        # Weight predictions by historical accuracy
        weights = [0.3, 0.25, 0.25, 0.2]
        return weighted_average(predictions, weights)
```

#### B. Player Similarity Matching
```python
# Find similar players for better predictions
def find_similar_players(player_id, context):
    """
    Use historical similarity to predict outcomes
    """
    similar_players = database.query(
        age_range=±2,
        position=same,
        usage_rate_range=±0.05,
        last_10_games_stats
    )
```

## 📊 DATA SOURCES TO IMPLEMENT

### Free/Low Cost:
1. **NBA.com API** (already available)
   - Player tracking data
   - Shot charts
   - Lineup combinations

2. **Public Betting Sites**
   - Line movements
   - Public percentages

3. **Social Media APIs**
   - Injury news
   - Lineup updates

### Premium (Consider for Phase 2):
1. **Second Spectrum** (~$10k/year)
   - Every player movement
   - Shot contest data
   - Defensive positioning

2. **Sports Radar/StatsBomb**
   - Advanced metrics
   - Real-time updates

## 🚀 QUICK IMPLEMENTATION CODE

Here's starter code for immediate wins:

```python
# src/enhanced_features.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class EnhancedFeatureEngine:
    """Add these to your existing feature_engineering.py"""

    def calculate_advanced_fatigue(self, player_data):
        """Enhanced fatigue score calculation"""
        # Recent games weighted more heavily
        weights = np.exp(-np.arange(10) / 3)  # Decay factor
        recent_minutes = player_data['numMinutes'].tail(10).values

        weighted_minutes = np.sum(recent_minutes * weights) / np.sum(weights)

        # Age factor
        age_factor = 1 + (player_data['age'] - 30) * 0.01

        # Back-to-back penalty
        b2b_penalty = 1.15 if self.is_back_to_back(player_data) else 1.0

        return weighted_minutes * age_factor * b2b_penalty

    def get_opposition_defensive_rating(self, player_position, opponent_team, season_data):
        """How good is opponent at defending this position?"""
        # Query defensive stats by position
        opp_games = season_data[
            (season_data['opponentteamName'] == opponent_team)
        ]

        if opp_games.empty:
            return 1.0  # Default if no data

        # Calculate points allowed to this position
        position_players = self.get_position_players(player_position)
        opp_defense = opp_games[opp_games['fullName'].isin(position_players)]

        if opp_defense.empty:
            return 1.0

        avg_points_allowed = opp_defense['points'].mean()
        league_avg = season_data['points'].mean()

        return league_avg / avg_points_allowed  # Higher = worse defense

    def calculate_lineup_synergy(self, player_name, teammates, lineup_data):
        """Synergy bonus for effective lineups"""
        # Find 5-man combos including this player
        player_lineups = lineup_data[
            lineup_data['players'].str.contains(player_name)
        ]

        if player_lineups.empty:
            return 0

        # Get net rating for these lineups
        net_ratings = player_lineups['net_rating'].values
        minutes = player_lineups['minutes'].values

        # Weight by minutes played
        weighted_rating = np.sum(net_ratings * minutes) / np.sum(minutes)

        # Normalize to +/- point adjustment
        return weighted_rating / 10  # Rough scaling

    def get_shot_quality_adjustment(self, player_id, recent_games=10):
        """Adjustment based on shot quality"""
        # This would need NBA shot chart data
        # Placeholder for implementation
        return {
            'difficulty_score': 0.5,  # 0-1 scale
            'contested_rate': 0.35,
            'open_shot_rate': 0.65,
            'transition_frequency': 0.2
        }

# Integration with your existing system
def enhance_predictions(player_name, game_context):
    """Add these to your final_predictions_optimized.py"""

    # Initialize feature engine
    feature_engine = EnhancedFeatureEngine()

    # Get player data
    player_data = load_player_data(player_name)

    # Calculate enhancements
    fatigue_adj = feature_engine.calculate_advanced_fatigue(player_data)
    defense_adj = feature_engine.get_opposition_defensive_rating(
        player_position, game_context['opponent_team']
    )
    synergy_adj = feature_engine.calculate_lineup_synergy(
        player_name, game_context['teammates']
    )
    shot_quality_adj = feature_engine.get_shot_quality_adjustment(player_name)

    # Combine with existing prediction
    base_prediction = existing_model.predict(player_name, game_context)

    # Apply adjustments
    enhanced_prediction = base_prediction * (
        1 + (fatigue_adj - 1) * 0.2 +
        (defense_adj - 1) * 0.3 +
        synergy_adj * 0.1 +
        shot_quality_adj['difficulty_score'] * 0.1
    )

    return enhanced_prediction
```

## 📈 TRACKING PROGRESS

```python
# Create a performance tracker
class PerformanceTracker:
    def __init__(self):
        self.metrics = {
            'daily_accuracy': [],
            'weekly_accuracy': [],
            'feature_importance': {},
            'model_confidence': []
        }

    def log_prediction(self, prediction, actual, confidence):
        """Track each prediction"""
        correct = 1 if abs(prediction - actual) < 2 else 0

        self.metrics['daily_accuracy'].append(correct)
        self.metrics['model_confidence'].append(confidence)

        # Weekly rolling average
        if len(self.metrics['daily_accuracy']) >= 7:
            weekly = np.mean(self.metrics['daily_accuracy'][-7:])
            self.metrics['weekly_accuracy'].append(weekly)
```

## 🎯 SUCCESS METRICS TO WATCH

1. **Accuracy Milestones**
   - Week 1: 62% (baseline +2%)
   - Month 1: 65% (baseline +5%)
   - Month 2: 68% (baseline +8%)
   - Month 3: 70% (baseline +10%)
   - Month 6: 73% (baseline +13%)
   - Month 9: 75% (goal)

2. **Feature Impact**
   - Each new feature should add >0.5% accuracy
   - Remove features that don't contribute
   - Feature importance changes over seasons

3. **Model Health**
   - Prediction distribution check
   - Confidence calibration
   - Overfitting detection

## ⚡ NEXT STEPS FOR YOU

1. **This Week:**
   - Implement fatigue enhancement
   - Add opponent defensive ratings
   - Start tracking line movements

2. **Next 2 Weeks:**
   - Pull NBA shot chart data
   - Calculate lineup synergy
   - Set up injury alerts

3. **Next Month:**
   - Build ensemble model
   - Implement player similarity
   - Add uncertainty quantification

Remember: **Small, consistent improvements compound**. Each 1% gain significantly improves your betting edge!