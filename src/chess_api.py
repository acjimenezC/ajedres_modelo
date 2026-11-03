"""
chess_api.py
============
Client for the Chess.com public API.

Provides methods to retrieve player profiles, game archives, and recent games.
Handles rate-limiting, missing users, and months without games gracefully.
"""

import re
import time
import requests
from typing import Optional

BASE_URL = "https://api.chess.com/pub"
HEADERS = {"User-Agent": "ChessPredictionBot/1.0 (academic project)"}
REQUEST_DELAY = 0.5  # seconds between requests


class ChessAPIClient:
    """
    Wrapper around the Chess.com public API.

    All methods return None or an empty list on error so callers
    never need to handle exceptions from network failures.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, url: str, retries: int = 3) -> Optional[dict]:
        """
        Perform a GET request with retry logic for 429 responses.

        Returns the parsed JSON dict on success, None on failure.
        """
        for attempt in range(retries):
            try:
                time.sleep(REQUEST_DELAY)
                resp = self.session.get(url, timeout=10)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 404:
                    return None  # User or resource not found
                elif resp.status_code == 429:
                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue
                else:
                    return None
            except requests.exceptions.Timeout:
                time.sleep(1)
                continue
            except requests.exceptions.RequestException:
                return None
        return None

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_player_profile(self, username: str) -> Optional[dict]:
        """
        Retrieve a player's public profile.

        Parameters
        ----------
        username : str
            Chess.com username.

        Returns
        -------
        dict or None
            Profile dict on success, None if user not found or error.
        """
        url = f"{BASE_URL}/player/{username.lower()}"
        return self._get(url)

    def get_player_rating(self, username: str, time_class: str) -> Optional[int]:
        """
        Retrieve a player's current rating for a given time class.

        Reads the /stats endpoint and extracts the ``last.rating`` of the
        sub-object matching the requested time class (chess_bullet,
        chess_blitz, chess_rapid or chess_daily).

        Parameters
        ----------
        username : str
            Chess.com username.
        time_class : str
            One of 'bullet', 'blitz', 'rapid', 'daily'.

        Returns
        -------
        int or None
            Current rating on success, None if the user, the time-class
            sub-object or the rating value is missing.
        """
        url = f"{BASE_URL}/player/{username.lower()}/stats"
        data = self._get(url)
        if data is None:
            return None

        key = f"chess_{str(time_class).lower()}"
        section = data.get(key)
        if not isinstance(section, dict):
            return None

        last = section.get("last")
        if not isinstance(last, dict):
            return None

        rating = last.get("rating")
        if rating is None:
            return None

        try:
            return int(rating)
        except (TypeError, ValueError):
            return None

    def get_player_archives(self, username: str) -> list:
        """
        Retrieve the list of monthly archive URLs for a player.

        Parameters
        ----------
        username : str
            Chess.com username.

        Returns
        -------
        list
            List of archive URL strings, newest last.  Empty on error.
        """
        url = f"{BASE_URL}/player/{username.lower()}/games/archives"
        data = self._get(url)
        if data is None:
            return []
        return data.get("archives", [])

    def get_monthly_games(self, username: str, year: int, month: int) -> list:
        """
        Retrieve all games for a player in a given month.

        Parameters
        ----------
        username : str
        year : int
        month : int

        Returns
        -------
        list
            List of game dicts.  Empty on error or no games.
        """
        url = f"{BASE_URL}/player/{username.lower()}/games/{year}/{month:02d}"
        data = self._get(url)
        if data is None:
            return []
        return data.get("games", [])

    def get_recent_games(
        self,
        username: str,
        time_class: str,
        n: int = 6,
    ) -> list:
        """
        Retrieve the n most recent completed games for a player of a
        specific time class (bullet, blitz, rapid).

        Games are returned in chronological order (oldest first).

        Parameters
        ----------
        username : str
        time_class : str
            One of 'bullet', 'blitz', 'rapid'.
        n : int
            Maximum number of games to return.

        Returns
        -------
        list
            Up to n game dicts enriched with parsed PGN fields.
        """
        archives = self.get_player_archives(username)
        if not archives:
            return []

        collected = []
        # Iterate archives newest-first
        for archive_url in reversed(archives):
            if len(collected) >= n:
                break
            # Extract year/month from archive URL
            parts = archive_url.rstrip("/").split("/")
            try:
                year = int(parts[-2])
                month = int(parts[-1])
            except (ValueError, IndexError):
                continue

            games = self.get_monthly_games(username, year, month)
            # Filter by time class and that the game is finished (has PGN)
            for game in reversed(games):
                if len(collected) >= n:
                    break
                if game.get("time_class", "").lower() != time_class.lower():
                    continue
                if not game.get("pgn"):
                    continue
                parsed = self._parse_game(game, username)
                if parsed is None:
                    continue
                collected.append(parsed)

        # Return in chronological order (oldest first)
        collected.reverse()
        return collected

    # ------------------------------------------------------------------
    # PGN parsing
    # ------------------------------------------------------------------

    def _parse_game(self, game: dict, username: str) -> Optional[dict]:
        """
        Enrich a raw game dict with fields extracted from the PGN.

        Returns None if the game is aborted or has an unknown result.
        """
        white = game.get("white", {})
        black = game.get("black", {})

        white_username = white.get("username", "").lower()
        black_username = black.get("username", "").lower()
        user_lower = username.lower()

        # Determine player's color
        if user_lower == white_username:
            color = "white"
        elif user_lower == black_username:
            color = "black"
        else:
            return None  # Player not in this game

        # Result from API (more reliable than PGN for Chess.com)
        if color == "white":
            result = white.get("result", "unknown")
            opponent_result = black.get("result", "unknown")
            player_rating = white.get("rating")
            opponent_rating = black.get("rating")
            opponent_username = black_username
        else:
            result = black.get("result", "unknown")
            opponent_result = white.get("result", "unknown")
            player_rating = black.get("rating")
            opponent_rating = white.get("rating")
            opponent_username = white_username

        # Skip aborted games
        if result == "abandoned" and opponent_result == "abandoned":
            return None
        if result in ("abandoned",):
            return None

        # Extract end_time for ordering (use epoch if available)
        end_time = game.get("end_time", 0)

        return {
            "username": username,
            "color": color,
            "result": result,          # 'win', 'resigned', 'checkmated', etc.
            "player_rating": player_rating,
            "opponent_rating": opponent_rating,
            "opponent_username": opponent_username,
            "time_class": game.get("time_class", ""),
            "end_time": end_time,
            "white_username": white_username,
            "black_username": black_username,
            "pgn": game.get("pgn", ""),
        }


# ------------------------------------------------------------------
# Module self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    client = ChessAPIClient()
    profile = client.get_player_profile("magnuscarlsen")
    if profile:
        print(f"Profile OK: {profile.get('username')}")
    else:
        print("Could not retrieve profile.")

    games = client.get_recent_games("magnuscarlsen", "blitz", n=3)
    print(f"Recent blitz games: {len(games)}")
