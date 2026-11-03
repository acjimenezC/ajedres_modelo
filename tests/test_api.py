"""
tests/test_api.py
=================
Tests for the Chess.com API client.

All tests run fully offline: every network call is mocked.
"""

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.chess_api import ChessAPIClient


class TestChessAPIClient:
    """Tests for ChessAPIClient (no network access)."""

    def setup_method(self):
        self.client = ChessAPIClient()

    # ------------------------------------------------------------------
    # Profile tests
    # ------------------------------------------------------------------

    def test_get_profile_valid(self):
        """A mocked _get payload should be returned as the profile dict."""
        payload = {"username": "magnuscarlsen", "player_id": 123}
        with patch.object(self.client, "_get", return_value=payload):
            profile = self.client.get_player_profile("magnuscarlsen")
        assert isinstance(profile, dict)
        assert profile["username"] == "magnuscarlsen"

    def test_get_profile_invalid(self):
        """A non-existent user (mocked None) must return None without raising."""
        with patch.object(self.client, "_get", return_value=None):
            profile = self.client.get_player_profile("this_user_does_not_exist_xyz")
        assert profile is None

    # ------------------------------------------------------------------
    # Current rating tests
    # ------------------------------------------------------------------

    def test_get_player_rating(self):
        """get_player_rating should read last.rating from the time-class stats."""
        stats = {"chess_blitz": {"last": {"rating": 2500}}}
        with patch.object(self.client, "_get", return_value=stats):
            rating = self.client.get_player_rating("magnuscarlsen", "blitz")
        assert rating == 2500

    def test_get_player_rating_miss(self):
        """A missing time-class sub-object must yield None."""
        with patch.object(self.client, "_get", return_value={"chess_rapid": {}}):
            rating = self.client.get_player_rating("magnuscarlsen", "blitz")
        assert rating is None

        with patch.object(self.client, "_get", return_value=None):
            rating = self.client.get_player_rating("nobody", "blitz")
        assert rating is None

    # ------------------------------------------------------------------
    # Recent games tests
    # ------------------------------------------------------------------

    def test_get_recent_games(self):
        """get_recent_games must return a list (archives mocked empty)."""
        with patch.object(self.client, "_get", return_value={"archives": []}):
            games = self.client.get_recent_games("magnuscarlsen", "blitz", n=3)
        assert isinstance(games, list)

    # ------------------------------------------------------------------
    # Rate limit handling
    # ------------------------------------------------------------------

    def test_rate_limit_handling(self):
        """A 429 response must be handled gracefully (return None, no exception)."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch.object(self.client.session, "get", return_value=mock_response):
            result = self.client._get("https://api.chess.com/pub/player/anyuser")

        assert result is None
