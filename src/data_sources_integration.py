"""
Data Sources Integration for NBA Betting Model
Integrates multiple APIs and data sources for enhanced predictions
Key sources: NBA API, betting odds APIs, injury reports
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import warnings
warnings.filterwarnings('ignore')

class NBADataIntegrator:
    """
    Integrates multiple data sources for enhanced predictions
    Focus on high-impact, accessible data sources
    """

    def __init__(self):
        self.api_base = "https://stats.nba.com/stats"
        self.headers = {
            'Accept': 'application/json',
            'Accept-Language': 'en-US',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache

    def _make_nba_request(self, endpoint, params=None):
        """
        Make request to NBA.com API with proper headers
        """
        url = f"{self.api_base}/{endpoint}"

        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching NBA data: {e}")
            return None

    def get_advanced_player_stats(self, player_id, season='2023-24'):
        """
        Get advanced statistics for a player
        Includes PER, TS%, Usage Rate, etc.
        """
        cache_key = f"advanced_stats_{player_id}_{season}"

        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                return cached_data

        params = {
            'PlayerID': player_id,
            'Season': season,
            'SeasonType': 'Regular Season',
            'MeasureType': 'Advanced',
            'PerMode': 'PerGame',
            'PlusMinus': 'N',
            'PaceAdjust': 'N',
            'Rank': 'N',
            'Outcome': '',
            'Location': '',
            'Month': '0',
            'SeasonSegment': '',
            'DateFrom': '',
            'DateTo': '',
            'OpponentTeamID': '0',
            'VsConference': '',
            'VsDivision': '',
            'GameSegment': '',
            'Period': '0',
            'ShotClockRange': '',
            'LastNGames': '0'
        }

        data = self._make_nba_request('playerdashboardbygeneralsplits', params)

        if data and 'resultSets' in data and data['resultSets']:
            result_set = data['resultSets'][0]
            if 'rowSet' in result_set and result_set['rowSet']:
                # Convert to DataFrame
                df = pd.DataFrame(result_set['rowSet'], columns=result_set['headers'])
                self.cache[cache_key] = (time.time(), df)
                return df

        return None

    def get_shot_chart_data(self, player_id, season='2023-24'):
        """
        Get shot chart data for shot quality analysis
        """
        cache_key = f"shotchart_{player_id}_{season}"

        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                return cached_data

        params = {
            'PlayerID': player_id,
            'Season': season,
            'SeasonType': 'Regular Season'
        }

        data = self._make_nba_request('shotchartdetail', params)

        if data and 'resultSets' in data and data['resultSets']:
            result_set = data['resultSets'][0]
            if 'rowSet' in result_set and result_set['rowSet']:
                df = pd.DataFrame(result_set['rowSet'], columns=result_set['headers'])
                self.cache[cache_key] = (time.time(), df)
                return df

        return None

    def get_lineup_data(self, team_id, season='2023-24'):
        """
        Get lineup combinations and their effectiveness
        """
        cache_key = f"lineups_{team_id}_{season}"

        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                return cached_data

        params = {
            'TeamID': team_id,
            'Season': season,
            'SeasonType': 'Regular Season',
            'PerMode': 'Per48',
            'GameSegment': '',
            'Period': '0',
            'LastNGames': '0',
            'MeasureType': 'Base',
            'PaceAdjust': 'N',
            'PlusMinus': 'Y',
            'OpponentTeamID': '0',
            'VsDivision': '',
            'VsConference': '',
            'GameScope': '',
            'PlayerPosition': '',
            'StarterBench': '',
            'GameSegment': '',
            'Month': '0',
            'SeasonSegment': '',
            'DateFrom': '',
            'DateTo': '',
            'Outcome': '',
            'Location': ''
        }

        data = self._make_nba_request('teamdashlineups', params)

        if data and 'resultSets' in data and data['resultSets']:
            result_set = data['resultSets'][0]
            if 'rowSet' in result_set and result_set['rowSet']:
                df = pd.DataFrame(result_set['rowSet'], columns=result_set['headers'])
                self.cache[cache_key] = (time.time(), df)
                return df

        return None

    def get_injury_report(self):
        """
        Get injury report from public sources
        Note: NBA doesn't have official injury API, so this uses common patterns
        """
        # This would need to integrate with sports news APIs
        # For now, returning structure for implementation

        injury_data = {
            'players': [
                {
                    'name': 'Player Name',
                    'status': 'questionable',  # probable/questionable/out
                    'injury': 'Ankle sprain',
                    'days_since': 3,
                    'impact': 0.85  # Performance reduction factor
                }
            ],
            'last_updated': datetime.now().isoformat()
        }

        return injury_data

    def calculate_shot_quality_metrics(self, shot_chart_df):
        """
        Calculate shot quality metrics from shot chart data
        """
        if shot_chart_df is None or len(shot_chart_df) == 0:
            return self._default_shot_quality()

        metrics = {}

        # Contested vs uncontested shots
        if 'CLOSE_DEF_DIST' in shot_chart_df.columns:
            contested_shots = shot_chart_df[shot_chart_df['CLOSE_DEF_DIST'] <= 4]
            total_shots = len(shot_chart_df)

            metrics['contested_shot_rate'] = len(contested_shots) / total_shots if total_shots > 0 else 0.5

            # Shooting efficiency by distance
            if 'SHOT_MADE_FLAG' in shot_chart_df.columns:
                contested_fg = contested_shots['SHOT_MADE_FLAG'].mean() if len(contested_shots) > 0 else 0.4
                open_shots = shot_chart_df[shot_chart_df['CLOSE_DEF_DIST'] > 4]
                open_fg = open_shots['SHOT_MADE_FLAG'].mean() if len(open_shots) > 0 else 0.5

                metrics['contested_fg_pct'] = contested_fg
                metrics['open_fg_pct'] = open_fg
                metrics['shot_quality_advantage'] = open_fg - contested_fg

        # Shot location distribution
        if 'SHOT_ZONE_BASIC' in shot_chart_df.columns:
            zone_counts = shot_chart_df['SHOT_ZONE_BASIC'].value_counts(normalize=True)
            metrics['corner_three_rate'] = zone_counts.get('Corner 3', 0)
            metrics['paint_rate'] = zone_counts.get('Restricted Area', 0)
            metrics['mid_range_rate'] = zone_counts.get('Mid-Range', 0)

        return metrics

    def get_opponent_defensive_ratings(self, opponent_team_id, season='2023-24'):
        """
        Get opponent defensive ratings by position
        """
        params = {
            'TeamID': opponent_team_id,
            'Season': season,
            'SeasonType': 'Regular Season',
            'MeasureType': 'Advanced',
            'PerMode': 'PerGame',
            'PlusMinus': 'N',
            'PaceAdjust': 'Y',
            'Rank': 'N'
        }

        data = self._make_nba_request('teamdashboardbygeneralsplits', params)

        if data and 'resultSets' in data and data['resultSets']:
            result_set = data['resultSets'][0]
            if 'rowSet' in result_set and result_set['rowSet']:
                df = pd.DataFrame(result_set['rowSet'], columns=result_set['headers'])

                # Extract defensive metrics
                if 'DEF_RATING' in df.columns:
                    defensive_rating = df['DEF_RATING'].iloc[0]
                    league_avg = 110  # Approximate league average

                    return {
                        'defensive_rating': defensive_rating,
                        'defensive_factor': league_avg / defensive_rating,
                        'pace_adjusted_defense': defensive_rating * (1 + df['PACE'].iloc[0] / 100)
                    }

        return self._default_defensive_rating()

    def get_betting_line_movement(self, player_name, market_type='points'):
        """
        Track line movements from betting sites
        Note: This would need integration with betting odds APIs
        """
        # Structure for implementation with odds API
        line_movement = {
            'opening_line': 22.5,
            'current_line': 23.0,
            'movement': 0.5,
            'movement_percentage': 2.2,  # (current - opening) / opening
            'time_opened': '2024-01-01T10:00:00',
            'sharp_movement': False,
            'public_percentage_over': 65
        }

        return line_movement

    def _default_shot_quality(self):
        """Default shot quality when no data available"""
        return {
            'contested_shot_rate': 0.35,
            'contested_fg_pct': 0.35,
            'open_fg_pct': 0.50,
            'shot_quality_advantage': 0.15,
            'corner_three_rate': 0.25,
            'paint_rate': 0.40,
            'mid_range_rate': 0.15
        }

    def _default_defensive_rating(self):
        """Default defensive rating when no data available"""
        return {
            'defensive_rating': 110,
            'defensive_factor': 1.0,
            'pace_adjusted_defense': 110
        }


class RealTimeDataFeed:
    """
    Real-time data feed for live updates
    """

    def __init__(self, update_interval=60):
        self.update_interval = update_interval
        self.last_update = {}
        self.subscribers = []

    def subscribe(self, callback):
        """Subscribe to real-time updates"""
        self.subscribers.append(callback)

    def check_for_updates(self):
        """Check for new data and notify subscribers"""
        # Check lineups
        new_lineups = self._get_latest_lineups()
        if new_lineups:
            for callback in self.subscribers:
                callback('lineups', new_lineups)

        # Check injuries
        new_injuries = self._get_latest_injuries()
        if new_injuries:
            for callback in self.subscribers:
                callback('injuries', new_injuries)

    def _get_latest_lineups(self):
        """Get latest lineup information"""
        # Implementation would check NBA.com or sports news APIs
        return None

    def _get_latest_injuries(self):
        """Get latest injury updates"""
        # Implementation would check sports news APIs
        return None


# Integration function for enhanced predictions
def enhance_predictions_with_new_data(player_name, game_context):
    """
    Enhance your predictions with new data sources
    """
    integrator = NBADataIntegrator()

    # Get player ID (you would need a mapping from name to ID)
    player_id = get_player_id(player_name)  # Implement this mapping

    enhanced_features = {}

    # 1. Advanced player stats
    advanced_stats = integrator.get_advanced_player_stats(player_id)
    if advanced_stats is not None:
        # Extract key metrics
        enhanced_features['player_efficiency_rating'] = advanced_stats.get('PER', 15.0)
        enhanced_features['true_shooting_pct'] = advanced_stats.get('TS_PCT', 0.55)
        enhanced_features['usage_rate'] = advanced_stats.get('USG_PCT', 0.20)
        enhanced_features['win_shares'] = advanced_stats.get('WS', 5.0)

    # 2. Shot quality metrics
    shot_chart = integrator.get_shot_chart_data(player_id)
    shot_quality = integrator.calculate_shot_quality_metrics(shot_chart)
    enhanced_features.update(shot_quality)

    # 3. Opponent defense
    opponent_id = get_team_id(game_context['opponent_team'])  # Implement this
    defense_data = integrator.get_opponent_defensive_ratings(opponent_id)
    enhanced_features.update(defense_data)

    # 4. Line movement data
    line_movement = integrator.get_betting_line_movement(player_name)
    enhanced_features['line_movement'] = line_movement['movement_percentage']
    enhanced_features['sharp_indicator'] = 1 if line_movement['sharp_movement'] else 0

    # 5. Injury impact
    injuries = integrator.get_injury_report()
    player_injury = next((i for i in injuries['players'] if i['name'] == player_name), None)
    if player_injury:
        enhanced_features['injury_impact'] = player_injury['impact']
        enhanced_features['injury_status'] = player_injury['status']
    else:
        enhanced_features['injury_impact'] = 1.0
        enhanced_features['injury_status'] = 'active'

    return enhanced_features


# Helper functions (implement these)
def get_player_id(player_name):
    """
    Map player name to NBA player ID
    You would implement this using a lookup table or API call
    """
    # Create a mapping or use NBA API to get ID
    player_id_map = {
        'LeBron James': 2544,
        'Stephen Curry': 201939,
        'Kevin Durant': 201142,
        # Add all players you track
    }
    return player_id_map.get(player_name, None)

def get_team_id(team_name):
    """
    Map team name to NBA team ID
    """
    team_id_map = {
        'Los Angeles Lakers': 1610612747,
        'Golden State Warriors': 1610612740,
        'Brooklyn Nets': 1610612741,
        # Add all teams
    }
    return team_id_map.get(team_name, None)