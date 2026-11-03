"""
data_collection.py
==================
Fetches player histories from Chess.com and assembles a training dataset.

Key design principles
---------------------
* All features for a given match are computed from games played BEFORE
  that match (reference_date parameter).  No future information leaks in.
* If a player has fewer games than the requested window, we use whatever
  is available and record games_available so models can weight accordingly.
* We never invent ratings or results.
"""

import os
import sys
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

# Allow running as script from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.chess_api import ChessAPIClient
from src.feature_engineering import FeatureEngineer
from src.preprocessing import DataPreprocessor, DRAW_RESULTS, LOSS_RESULTS

# --------------------------------------------------------------------------
# Known players for demo dataset collection
# --------------------------------------------------------------------------
DEMO_PLAYERS = [
    "magnuscarlsen",
    "hikaru",
    "fabianocaruana",
    "anishgiri",
    "theevildoer",
    "polski_filipek",
    "firouzja2003",
    "penguingm1",
]


class DataCollector:
    """
    Builds the training dataset from Chess.com game histories.
    """

    def __init__(self, window: int = 6):
        self.api = ChessAPIClient()
        self.fe = FeatureEngineer()
        self.pre = DataPreprocessor()
        self.default_window = window

    # ------------------------------------------------------------------
    # Per-player feature extraction
    # ------------------------------------------------------------------

    def collect_player_features(
        self,
        username: str,
        color: str,
        time_class: str,
        window: int = None,
        reference_date: Optional[datetime] = None,
    ) -> dict:
        """
        Compute pre-match feature statistics for one player.

        Only games completed BEFORE reference_date are used.
        If reference_date is None, all available games are used.

        Parameters
        ----------
        username : str
        color : str
            'white' or 'black' (the color this player will play in the match).
        time_class : str
        window : int
            How many recent games to use (5 or 6).
        reference_date : datetime, optional
            Cut-off date.  Games at or after this date are excluded.

        Returns
        -------
        dict
            Keys: rating, win_rate, loss_rate, draw_rate, win_rate_as_white,
            win_rate_as_black, avg_opponent_rating, rating_trend,
            games_available, recent_results.
        """
        if window is None:
            window = self.default_window

        # Retrieve recent games
        raw_games = self.api.get_recent_games(username, time_class, n=window * 3)

        # Filter by reference_date if supplied
        if reference_date is not None:
            ref_ts = reference_date.timestamp()
            raw_games = [g for g in raw_games if g.get("end_time", 0) < ref_ts]

        # Keep only the most recent `window` games
        raw_games = raw_games[-window:]

        games_available = len(raw_games)

        # Default (empty-history) values
        defaults = {
            "rating": None,
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "draw_rate": 0.0,
            "win_rate_as_white": 0.0,
            "win_rate_as_black": 0.0,
            "avg_opponent_rating": None,
            "rating_trend": 0.0,
            "games_available": 0,
            "recent_results": [],
        }

        if games_available == 0:
            return defaults

        # Get current rating from the last game in history
        current_rating = None
        if raw_games:
            current_rating = raw_games[-1].get("player_rating")

        # Per-game stats
        wins = 0
        losses = 0
        draws = 0
        white_games = 0
        white_wins = 0
        black_games = 0
        black_wins = 0
        opp_ratings = []
        recent_results = []

        for g in raw_games:
            result_label = self.pre.parse_game_result(g, username)
            recent_results.append(result_label)

            if result_label == "win":
                wins += 1
            elif result_label == "loss":
                losses += 1
            elif result_label == "draw":
                draws += 1

            g_color = g.get("color", "").lower()
            if g_color == "white":
                white_games += 1
                if result_label == "win":
                    white_wins += 1
            elif g_color == "black":
                black_games += 1
                if result_label == "win":
                    black_wins += 1

            opp_r = g.get("opponent_rating")
            if opp_r is not None:
                opp_ratings.append(opp_r)

        n = games_available
        win_rate = wins / n
        loss_rate = losses / n
        draw_rate = draws / n
        win_rate_as_white = white_wins / white_games if white_games > 0 else 0.0
        win_rate_as_black = black_wins / black_games if black_games > 0 else 0.0
        avg_opp_rating = float(np.mean(opp_ratings)) if opp_ratings else None

        # Rating trend: last game rating minus first game rating in window
        first_rating = raw_games[0].get("player_rating")
        last_rating = raw_games[-1].get("player_rating")
        if first_rating is not None and last_rating is not None:
            rating_trend = float(last_rating) - float(first_rating)
        else:
            rating_trend = 0.0

        return {
            "rating": current_rating,
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "draw_rate": draw_rate,
            "win_rate_as_white": win_rate_as_white,
            "win_rate_as_black": win_rate_as_black,
            "avg_opponent_rating": avg_opp_rating,
            "rating_trend": rating_trend,
            "games_available": games_available,
            "recent_results": recent_results,
        }

    # ------------------------------------------------------------------
    # Dataset builder
    # ------------------------------------------------------------------

    def build_dataset_from_players(
        self,
        player_pairs: List[dict],
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Build a labelled dataset from a list of (player_a, player_b, game) dicts.

        Each element of player_pairs must have:
            player_a, player_b, color_a, time_class, target (0/1/2)

        Parameters
        ----------
        player_pairs : list of dict
        save_path : str, optional
            If provided, saves the DataFrame as CSV.

        Returns
        -------
        pd.DataFrame
        """
        rows = []
        for pair in player_pairs:
            username_a = pair["player_a"]
            username_b = pair["player_b"]
            color_a = pair.get("color_a", "white")
            time_class = pair.get("time_class", "blitz")
            reference_date = pair.get("reference_date", None)
            target = pair.get("target", None)
            window = pair.get("window", self.default_window)

            feat_a = self.collect_player_features(
                username_a, color_a, time_class, window, reference_date
            )
            color_b = "black" if color_a == "white" else "white"
            feat_b = self.collect_player_features(
                username_b, color_b, time_class, window, reference_date
            )

            row = self.fe.build_match_features(feat_a, feat_b, color_a, time_class)
            row["target"] = target
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"Dataset saved to {save_path}")

        return df

    # ------------------------------------------------------------------
    # Helper: build training rows from head-to-head real games
    # ------------------------------------------------------------------

    def collect_head_to_head_dataset(
        self,
        players: List[str],
        time_class: str = "blitz",
        window: int = 6,
        games_per_player: int = 20,
    ) -> pd.DataFrame:
        """
        For each player in `players`, fetch their recent games and for each
        completed game construct a training row (using only pre-game info).

        Parameters
        ----------
        players : list of str
        time_class : str
        window : int
        games_per_player : int

        Returns
        -------
        pd.DataFrame
        """
        rows = []

        for username in players:
            print(f"  Collecting games for {username}...")
            raw_games = self.api.get_recent_games(
                username, time_class, n=games_per_player
            )

            for i, game in enumerate(raw_games):
                # reference_date = just before this game started
                ref_ts = game.get("end_time", 0)
                if ref_ts == 0:
                    continue

                ref_date = datetime.fromtimestamp(ref_ts, tz=timezone.utc)
                # Use games before this one as context
                games_before = raw_games[:i]

                opp_username = game.get("opponent_username", "")
                if not opp_username:
                    continue

                player_color = game.get("color", "white")
                result_label = self.pre.parse_game_result(game, username)
                if result_label == "unknown":
                    continue

                target = self.pre.encode_result_for_target(result_label)

                # Build features from existing history (games_before)
                feat_a = self._features_from_games(
                    games_before[-window:], game.get("player_rating")
                )
                # Fetch opponent's history as of ref_date
                feat_b = self.collect_player_features(
                    opp_username, "black" if player_color == "white" else "white",
                    time_class, window, ref_date
                )

                row = self.fe.build_match_features(
                    feat_a, feat_b, player_color, time_class
                )
                row["target"] = target
                rows.append(row)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def _features_from_games(self, games: list, current_rating) -> dict:
        """Build a features dict from a pre-filtered list of games."""
        if not games:
            return {
                "rating": current_rating,
                "win_rate": 0.0,
                "loss_rate": 0.0,
                "draw_rate": 0.0,
                "win_rate_as_white": 0.0,
                "win_rate_as_black": 0.0,
                "avg_opponent_rating": current_rating,
                "rating_trend": 0.0,
                "games_available": 0,
            }

        wins = losses = draws = 0
        white_games = white_wins = black_games = black_wins = 0
        opp_ratings = []

        for g in games:
            result_code = g.get("result", "unknown").lower()
            if result_code == "win":
                outcome = "win"
            elif result_code in DRAW_RESULTS:
                outcome = "draw"
            elif result_code in LOSS_RESULTS:
                outcome = "loss"
            else:
                continue

            if outcome == "win":
                wins += 1
            elif outcome == "loss":
                losses += 1
            else:
                draws += 1

            g_color = g.get("color", "").lower()
            if g_color == "white":
                white_games += 1
                if outcome == "win":
                    white_wins += 1
            elif g_color == "black":
                black_games += 1
                if outcome == "win":
                    black_wins += 1

            opp_r = g.get("opponent_rating")
            if opp_r:
                opp_ratings.append(opp_r)

        n = len(games)
        first_r = games[0].get("player_rating")
        last_r = games[-1].get("player_rating")
        trend = 0.0
        if first_r and last_r:
            trend = float(last_r) - float(first_r)

        return {
            "rating": current_rating,
            "win_rate": wins / n,
            "loss_rate": losses / n,
            "draw_rate": draws / n,
            "win_rate_as_white": white_wins / white_games if white_games else 0.0,
            "win_rate_as_black": black_wins / black_games if black_games else 0.0,
            "avg_opponent_rating": float(np.mean(opp_ratings)) if opp_ratings else current_rating,
            "rating_trend": trend,
            "games_available": n,
        }


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    collector = DataCollector(window=6)
    time_class = "blitz"

    print("Collecting head-to-head dataset from known players...")
    df = collector.collect_head_to_head_dataset(
        players=DEMO_PLAYERS,
        time_class=time_class,
        window=6,
        games_per_player=20,
    )

    if df.empty:
        print("No data collected. Check API availability.")
    else:
        print(f"Collected {len(df)} rows.")

        raw_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "raw", "games_raw.csv"
        )
        processed_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "processed", "dataset.csv"
        )

        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)

        df.to_csv(raw_path, index=False)
        df.dropna(subset=["target"]).to_csv(processed_path, index=False)
        print(f"Raw data: {raw_path}")
        print(f"Processed data: {processed_path}")
