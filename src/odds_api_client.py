"""
NBA Odds API Client for The Odds API

This module fetches live NBA player prop over/under odds and integrates with our ML models
for automated betting predictions.

Features:
- Fetches live NBA games and player props
- Filters for tracked players in our model
- Formats data for ML pipeline integration
- Handles API rate limiting and error management
"""

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os
import json
import warnings
warnings.filterwarnings('ignore')

class NBAOddsClient:
    """Client for fetching NBA odds from The Odds API."""

    def __init__(self, api_key=None):
        """Initialize the odds API client.

        Args:
            api_key (str): The Odds API key. If None, will try to read from environment.
        """
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError("API key required. Set ODDS_API_KEY environment variable or pass api_key parameter.")

        self.base_url = "https://api.the-odds-api.com/v4"
        self.sport = "basketball_nba"
        self.regions = "us"  # US sportsbooks only
        self.markets = "player_points,player_rebounds,player_assists"  # Main player prop markets

        # Load our tracked players
        self.tracked_players = self._load_tracked_players()

    def _load_tracked_players(self):
        """Load the list of players we track from our feature data."""
        try:
            # Load from our engineered features to get current player list
            data = pd.read_csv('../data/processed/engineered_features.csv')
            players = data['fullName'].unique().tolist()
            print(f"✓ Loaded {len(players)} tracked players from our model")
            return players
        except Exception as e:
            print(f"Warning: Could not load tracked players: {e}")
            return []

    def _make_request(self, endpoint, params=None):
        """Make API request with error handling and rate limiting."""
        url = f"{self.base_url}/{endpoint}"

        # Add API key to params
        if params is None:
            params = {}
        params['apiKey'] = self.api_key

        try:
            print(f"Making request to: {endpoint}")
            response = requests.get(url, params=params, timeout=30)

            # Check rate limit headers
            if 'x-requests-remaining' in response.headers:
                remaining = response.headers['x-requests-remaining']
                print(f"  API requests remaining: {remaining}")

                # If running low on requests, warn user
                if int(remaining) < 10:
                    print(f"⚠️  Warning: Only {remaining} API requests remaining!")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            return None

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse API response: {e}")
            return None

    def get_todays_nba_games(self):
        """Get today's NBA games with basic odds."""
        print("\n=== FETCHING TODAY'S NBA GAMES ===")

        params = {
            'regions': self.regions,
            'markets': 'h2h',  # Just get basic game info first
            'dateFormat': 'iso'
        }

        games = self._make_request(f"sports/{self.sport}/odds", params)

        if not games:
            print("❌ No games data received")
            return []

        print(f"✓ Found {len(games)} NBA games today")

        # Filter for games happening today or soon
        today = datetime.now().date()
        filtered_games = []

        for game in games:
            game_time = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))
            game_date = game_time.date()

            # Include games from today and tomorrow (for late night games)
            if game_date >= today and game_date <= today + timedelta(days=1):
                filtered_games.append({
                    'id': game['id'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'commence_time': game['commence_time'],
                    'game_time': game_time
                })

        print(f"✓ Filtered to {len(filtered_games)} relevant games")
        return filtered_games

    def get_player_props_for_game(self, game_id):
        """Get player props for a specific game."""
        print(f"\nFetching player props for game: {game_id}")

        params = {
            'regions': self.regions,
            'markets': self.markets,
            'dateFormat': 'iso'
        }

        # For player props, we need to use the events endpoint
        props = self._make_request(f"sports/{self.sport}/events/{game_id}/odds", params)

        if not props:
            print(f"❌ No player props found for game {game_id}")
            return []

        player_lines = []

        # Parse player props from the response
        for bookmaker in props.get('bookmakers', []):
            bookmaker_name = bookmaker['title']

            for market in bookmaker.get('markets', []):
                market_type = market['key']  # player_points, player_rebounds, etc.

                for outcome in market.get('outcomes', []):
                    player_name = outcome.get('description', '')
                    line_value = outcome.get('point')
                    over_odds = outcome.get('price')

                    if player_name and line_value and over_odds:
                        # Check if this player is in our tracked list
                        if any(tracked.lower() in player_name.lower() or
                              player_name.lower() in tracked.lower()
                              for tracked in self.tracked_players):

                            player_lines.append({
                                'game_id': game_id,
                                'player_name': player_name,
                                'market_type': market_type,
                                'line_value': line_value,
                                'over_odds': over_odds,
                                'bookmaker': bookmaker_name,
                                'timestamp': datetime.now()
                            })

        print(f"✓ Found {len(player_lines)} relevant player prop lines")
        return player_lines

    def get_all_todays_player_props(self):
        """Get player props for all of today's NBA games."""
        print("\n=== FETCHING ALL PLAYER PROPS FOR TODAY ===")

        # Get today's games
        games = self.get_todays_nba_games()

        if not games:
            print("❌ No games found for today")
            return pd.DataFrame()

        all_props = []

        for i, game in enumerate(games, 1):
            print(f"\n[{i}/{len(games)}] {game['away_team']} @ {game['home_team']}")

            # Get player props for this game
            props = self.get_player_props_for_game(game['id'])

            # Add game context to each prop
            for prop in props:
                prop.update({
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'game_time': game['game_time']
                })

            all_props.extend(props)

            # Small delay to be respectful of API rate limits
            time.sleep(0.5)

        if not all_props:
            print("❌ No player props found for tracked players")
            return pd.DataFrame()

        # Convert to DataFrame
        props_df = pd.DataFrame(all_props)

        print(f"\n✅ Successfully fetched {len(props_df)} player prop lines")
        print(f"✅ Covering {props_df['player_name'].nunique()} unique players")
        print(f"✅ Markets: {', '.join(props_df['market_type'].unique())}")

        return props_df

    def format_for_ml_pipeline(self, props_df, game_date=None):
        """Format the props data for integration with our ML pipeline."""
        print("\n=== FORMATTING DATA FOR ML PIPELINE ===")

        if props_df.empty:
            print("❌ No props data to format")
            return pd.DataFrame()

        # Use today's date if not specified
        if game_date is None:
            game_date = datetime.now().date()

        formatted_data = []

        for _, prop in props_df.iterrows():
            # Map market types to our model's expected format
            if prop['market_type'] == 'player_points':
                stat_type = 'points'
            elif prop['market_type'] == 'player_rebounds':
                stat_type = 'rebounds'
            elif prop['market_type'] == 'player_assists':
                stat_type = 'assists'
            else:
                continue  # Skip unknown market types

            # Create a record that matches our ML input format
            formatted_record = {
                'gameDate': game_date,
                'fullName': prop['player_name'],
                'home_team': prop['home_team'],
                'away_team': prop['away_team'],
                'game_time': prop['game_time'],
                'prop_line': prop['line_value'],
                'over_odds': prop['over_odds'],
                'bookmaker': prop['bookmaker'],
                'market_type': stat_type,
                'api_timestamp': prop['timestamp']
            }

            formatted_data.append(formatted_record)

        formatted_df = pd.DataFrame(formatted_data)

        print(f"✅ Formatted {len(formatted_df)} prop lines for ML pipeline")
        return formatted_df

def main():
    """Main function to test the odds API client."""
    print("=== NBA ODDS API CLIENT TEST ===")

    # Check for API key
    api_key = os.getenv('ODDS_API_KEY')
    if not api_key:
        print("\n❌ No API key found!")
        print("To use this module:")
        print("1. Sign up at https://the-odds-api.com/")
        print("2. Get your API key")
        print("3. Set environment variable: export ODDS_API_KEY='your_key_here'")
        return None

    try:
        # Initialize client
        client = NBAOddsClient(api_key)

        # Fetch today's player props
        props_df = client.get_all_todays_player_props()

        if not props_df.empty:
            # Format for ML pipeline
            formatted_df = client.format_for_ml_pipeline(props_df)

            # Save the data
            output_file = '../data/processed/todays_player_props.csv'
            formatted_df.to_csv(output_file, index=False)
            print(f"\n✅ Saved today's props to: {output_file}")

            # Display sample
            print("\n=== SAMPLE PROPS DATA ===")
            print(formatted_df.head(10).to_string(index=False))

            return formatted_df
        else:
            print("\n❌ No props data available")
            return pd.DataFrame()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

if __name__ == "__main__":
    main()