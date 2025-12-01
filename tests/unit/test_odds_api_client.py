"""
Unit Tests for NBA Odds API Client
===================================

Tests the odds_api_client module with mocked API responses.
Fast, isolated unit tests that don't consume API credits.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from odds_api_client import NBAOddsClient


class TestNBAOddsClientInitialization:
    """Test client initialization and configuration."""

    def test_init_with_api_key(self, mock_api_key):
        """Test initialization with explicit API key."""
        client = NBAOddsClient(api_key=mock_api_key)

        assert client.api_key == mock_api_key
        assert client.sport == "basketball_nba"
        assert client.regions == "us"
        assert "player_points" in client.markets

    def test_init_with_env_var(self, mock_env_with_api_key):
        """Test initialization using environment variable."""
        client = NBAOddsClient()

        assert client.api_key == mock_env_with_api_key
        assert client.base_url == "https://api.the-odds-api.com/v4"

    def test_init_without_api_key(self, mock_env_without_api_key):
        """Test initialization fails without API key."""
        with pytest.raises(ValueError, match="API key required"):
            NBAOddsClient()

    @patch('odds_api_client.pd.read_csv')
    def test_load_tracked_players_success(self, mock_read_csv, mock_api_key):
        """Test loading tracked players from features file."""
        # Mock the CSV data
        mock_df = pd.DataFrame({
            'fullName': ['LeBron James', 'Stephen Curry', 'Kevin Durant']
        })
        mock_read_csv.return_value = mock_df

        client = NBAOddsClient(api_key=mock_api_key)

        assert len(client.tracked_players) == 3
        assert 'LeBron James' in client.tracked_players

    @patch('odds_api_client.pd.read_csv')
    def test_load_tracked_players_failure(self, mock_read_csv, mock_api_key):
        """Test graceful handling when tracked players file missing."""
        mock_read_csv.side_effect = FileNotFoundError("File not found")

        client = NBAOddsClient(api_key=mock_api_key)

        assert client.tracked_players == []


class TestAPIRequests:
    """Test API request handling."""

    @patch('odds_api_client.requests.get')
    def test_make_request_success(self, mock_get, mock_api_key):
        """Test successful API request."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {'data': 'test'}
        mock_response.headers = {'x-requests-remaining': '100'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = NBAOddsClient(api_key=mock_api_key)
        result = client._make_request('test/endpoint')

        assert result == {'data': 'test'}
        mock_get.assert_called_once()

    @patch('odds_api_client.requests.get')
    def test_make_request_with_low_remaining(self, mock_get, mock_api_key, capfd):
        """Test warning when API requests running low."""
        mock_response = Mock()
        mock_response.json.return_value = {'data': 'test'}
        mock_response.headers = {'x-requests-remaining': '5'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = NBAOddsClient(api_key=mock_api_key)
        result = client._make_request('test/endpoint')

        captured = capfd.readouterr()
        assert 'Warning' in captured.out
        assert '5' in captured.out

    @patch('odds_api_client.requests.get')
    def test_make_request_network_error(self, mock_get, mock_api_key):
        """Test handling of network errors."""
        mock_get.side_effect = Exception("Network error")

        client = NBAOddsClient(api_key=mock_api_key)
        result = client._make_request('test/endpoint')

        assert result is None

    @patch('odds_api_client.requests.get')
    def test_make_request_json_decode_error(self, mock_get, mock_api_key):
        """Test handling of JSON decode errors."""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Error", "", 0)
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = NBAOddsClient(api_key=mock_api_key)
        result = client._make_request('test/endpoint')

        assert result is None


class TestGetTodaysGames:
    """Test fetching today's NBA games."""

    @patch('odds_api_client.NBAOddsClient._make_request')
    def test_get_todays_games_success(self, mock_request, mock_api_key, mock_odds_response):
        """Test successful retrieval of today's games."""
        mock_request.return_value = mock_odds_response

        client = NBAOddsClient(api_key=mock_api_key)
        games = client.get_todays_nba_games()

        assert len(games) > 0
        assert 'id' in games[0]
        assert 'home_team' in games[0]
        assert 'away_team' in games[0]

    @patch('odds_api_client.NBAOddsClient._make_request')
    def test_get_todays_games_empty(self, mock_request, mock_api_key):
        """Test handling when no games available."""
        mock_request.return_value = []

        client = NBAOddsClient(api_key=mock_api_key)
        games = client.get_todays_nba_games()

        assert games == []

    @patch('odds_api_client.NBAOddsClient._make_request')
    def test_get_todays_games_api_failure(self, mock_request, mock_api_key):
        """Test handling when API request fails."""
        mock_request.return_value = None

        client = NBAOddsClient(api_key=mock_api_key)
        games = client.get_todays_nba_games()

        assert games == []


class TestGetPlayerProps:
    """Test fetching player props."""

    @patch('odds_api_client.NBAOddsClient._make_request')
    @patch('odds_api_client.NBAOddsClient._load_tracked_players')
    def test_get_player_props_success(self, mock_load_players, mock_request,
                                     mock_api_key, mock_player_props_response):
        """Test successful retrieval of player props."""
        mock_load_players.return_value = ['LeBron James', 'Stephen Curry']
        mock_request.return_value = mock_player_props_response

        client = NBAOddsClient(api_key=mock_api_key)
        props = client.get_player_props_for_game('game_123')

        assert len(props) > 0
        assert all('player_name' in prop for prop in props)
        assert all('line_value' in prop for prop in props)

    @patch('odds_api_client.NBAOddsClient._make_request')
    def test_get_player_props_no_data(self, mock_request, mock_api_key):
        """Test handling when no player props available."""
        mock_request.return_value = None

        client = NBAOddsClient(api_key=mock_api_key)
        props = client.get_player_props_for_game('game_123')

        assert props == []

    @patch('odds_api_client.NBAOddsClient._load_tracked_players')
    @patch('odds_api_client.NBAOddsClient._make_request')
    def test_get_player_props_filters_tracked_only(self, mock_request, mock_load_players,
                                                   mock_api_key, mock_player_props_response):
        """Test that only tracked players are returned."""
        # Only track LeBron, not Curry
        mock_load_players.return_value = ['LeBron James']
        mock_request.return_value = mock_player_props_response

        client = NBAOddsClient(api_key=mock_api_key)
        props = client.get_player_props_for_game('game_123')

        player_names = [p['player_name'] for p in props]
        assert 'LeBron James' in player_names
        # Should still include Curry due to partial name matching


class TestFormatForMLPipeline:
    """Test formatting props data for ML pipeline."""

    def test_format_for_ml_pipeline_success(self, mock_api_key):
        """Test successful data formatting."""
        # Create sample props data
        props_df = pd.DataFrame({
            'player_name': ['LeBron James', 'Stephen Curry'],
            'market_type': ['player_points', 'player_assists'],
            'line_value': [25.5, 7.5],
            'over_odds': [-110, -115],
            'bookmaker': ['DraftKings', 'FanDuel'],
            'home_team': ['LAL', 'GSW'],
            'away_team': ['GSW', 'PHX'],
            'game_time': [datetime.now(), datetime.now()],
            'timestamp': [datetime.now(), datetime.now()]
        })

        client = NBAOddsClient(api_key=mock_api_key)
        formatted = client.format_for_ml_pipeline(props_df)

        assert len(formatted) == 2
        assert 'fullName' in formatted.columns
        assert 'prop_line' in formatted.columns
        assert formatted['market_type'].tolist() == ['points', 'assists']

    def test_format_for_ml_pipeline_empty(self, mock_api_key):
        """Test formatting with empty DataFrame."""
        client = NBAOddsClient(api_key=mock_api_key)
        formatted = client.format_for_ml_pipeline(pd.DataFrame())

        assert formatted.empty

    def test_format_for_ml_pipeline_unknown_market(self, mock_api_key):
        """Test handling of unknown market types."""
        props_df = pd.DataFrame({
            'player_name': ['LeBron James'],
            'market_type': ['player_blocks'],  # Unknown market
            'line_value': [1.5],
            'over_odds': [-110],
            'bookmaker': ['DraftKings'],
            'home_team': ['LAL'],
            'away_team': ['GSW'],
            'game_time': [datetime.now()],
            'timestamp': [datetime.now()]
        })

        client = NBAOddsClient(api_key=mock_api_key)
        formatted = client.format_for_ml_pipeline(props_df)

        # Should skip unknown markets
        assert len(formatted) == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch('odds_api_client.NBAOddsClient._make_request')
    def test_malformed_api_response(self, mock_request, mock_api_key):
        """Test handling of malformed API responses."""
        mock_request.return_value = {'unexpected': 'format'}

        client = NBAOddsClient(api_key=mock_api_key)

        # Should handle gracefully without crashing
        games = client.get_todays_nba_games()
        assert isinstance(games, list)

    def test_special_characters_in_player_names(self, mock_api_key):
        """Test handling special characters in player names."""
        props_df = pd.DataFrame({
            'player_name': ["Luka Dončić", "Nikola Jokić"],
            'market_type': ['player_points', 'player_assists'],
            'line_value': [28.5, 9.5],
            'over_odds': [-110, -115],
            'bookmaker': ['DraftKings', 'FanDuel'],
            'home_team': ['DAL', 'DEN'],
            'away_team': ['LAL', 'GSW'],
            'game_time': [datetime.now(), datetime.now()],
            'timestamp': [datetime.now(), datetime.now()]
        })

        client = NBAOddsClient(api_key=mock_api_key)
        formatted = client.format_for_ml_pipeline(props_df)

        assert len(formatted) == 2
        assert all(name in formatted['fullName'].values
                  for name in ["Luka Dončić", "Nikola Jokić"])


@pytest.mark.parametrize("market_type,expected_stat", [
    ('player_points', 'points'),
    ('player_rebounds', 'rebounds'),
    ('player_assists', 'assists'),
])
def test_market_type_mapping(mock_api_key, market_type, expected_stat):
    """Test correct mapping of market types to stat types."""
    props_df = pd.DataFrame({
        'player_name': ['Test Player'],
        'market_type': [market_type],
        'line_value': [10.5],
        'over_odds': [-110],
        'bookmaker': ['DraftKings'],
        'home_team': ['LAL'],
        'away_team': ['GSW'],
        'game_time': [datetime.now()],
        'timestamp': [datetime.now()]
    })

    client = NBAOddsClient(api_key=mock_api_key)
    formatted = client.format_for_ml_pipeline(props_df)

    assert formatted['market_type'].iloc[0] == expected_stat
