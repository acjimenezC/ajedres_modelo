"""
data_loader.py
==============
Downloads and processes the Kaggle Chess.com games dataset into the
Option-A training table.

The raw source is adityajha1504/chesscom-user-games-60000-games
(club_games_data.csv).  This module filters standard-chess games, derives
the 3-class target (0=win_white, 1=win_black, 2=draw), parses the
time_control string into base/increment seconds, encodes time_class, and
writes the processed feature table to data/processed/dataset.csv.
"""

import glob
import os

import kagglehub
import pandas as pd

# Allow running as a script from the project root
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.feature_engineering import (
    FeatureEngineer,
    TIME_CLASS_MAP,
    TARGET_WIN_WHITE,
    TARGET_WIN_BLACK,
    TARGET_DRAW,
)
from src.preprocessing import DRAW_RESULTS

KAGGLE_DATASET = "adityajha1504/chesscom-user-games-60000-games"
CSV_FILENAME = "club_games_data.csv"

# Project root resolved relative to this file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_time_control(tc) -> tuple:
    """
    Parse a Chess.com time_control string into base and increment seconds.

    Rules
    -----
    * If '+' is present, split on '+' into (base, increment).
    * Else if '/' is present, take the part AFTER '/' as base, increment 0.
    * Else the whole string is the base, increment 0.

    Parameters
    ----------
    tc : str
        Raw time_control value, e.g. '60', '180+2', '1/259200'.

    Returns
    -------
    tuple of int
        (base_seconds, increment_seconds).

    Examples
    --------
    >>> parse_time_control('60')
    (60, 0)
    >>> parse_time_control('180+2')
    (180, 2)
    >>> parse_time_control('1/259200')
    (259200, 0)
    """
    tc = str(tc)
    if "+" in tc:
        base, increment = tc.split("+", 1)
        return int(base), int(increment)
    if "/" in tc:
        base = tc.split("/", 1)[1]
        return int(base), 0
    return int(tc), 0


def download_dataset() -> str:
    """
    Download (or locate the cached copy of) the Kaggle dataset CSV.

    Returns
    -------
    str
        Absolute path to club_games_data.csv.

    Raises
    ------
    FileNotFoundError
        If the CSV cannot be found inside the downloaded directory.
    """
    path = kagglehub.dataset_download(KAGGLE_DATASET)

    # kagglehub returns a directory; locate the CSV inside it.
    if os.path.isfile(path):
        return path

    matches = glob.glob(os.path.join(path, "**", CSV_FILENAME), recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"{CSV_FILENAME} not found in downloaded dataset at {path}"
        )
    return matches[0]


def _derive_target(row) -> int:
    """
    Derive the 3-class target from the two result columns.

    Draws are detected first via either result column; otherwise the
    player whose result is 'win' determines the winner.  Rows that fit
    none of these are marked -1 for dropping.
    """
    white_result = str(row["white_result"]).lower()
    black_result = str(row["black_result"]).lower()

    if white_result in DRAW_RESULTS or black_result in DRAW_RESULTS:
        return TARGET_DRAW
    if white_result == "win":
        return TARGET_WIN_WHITE
    if black_result == "win":
        return TARGET_WIN_BLACK
    return -1


def _termination_reason(row) -> str:
    """Return the non-'win' result string, kept for future win-reason work."""
    white_result = str(row["white_result"]).lower()
    black_result = str(row["black_result"]).lower()
    if white_result != "win":
        return white_result
    return black_result


def build_processed_dataset(
    csv_path: str = None,
    save_path: str = "data/processed/dataset.csv",
) -> pd.DataFrame:
    """
    Build the Option-A processed dataset from the raw Kaggle CSV.

    Parameters
    ----------
    csv_path : str, optional
        Path to the raw CSV.  If None, download_dataset() is called.
    save_path : str
        Processed dataset output path (resolved relative to project root).

    Returns
    -------
    pd.DataFrame
        The processed feature table (6 model features + base_time + increment
        + target + termination_reason). base_time/increment are kept in the
        CSV for future use but are NOT model features.
    """
    if csv_path is None:
        csv_path = download_dataset()

    df = pd.read_csv(csv_path)
    print(f"Raw rows loaded: {len(df)}")

    # Keep only standard chess games.
    df = df[df["rules"] == "chess"].copy()
    print(f"Rows after rules=='chess' filter: {len(df)}")

    # Derive target and termination reason.
    df["target"] = df.apply(_derive_target, axis=1)
    df["termination_reason"] = df.apply(_termination_reason, axis=1)

    # Drop rows whose outcome could not be derived.
    df = df[df["target"] != -1].copy()
    print(f"Rows after dropping undecidable targets: {len(df)}")

    # Parse time control into base/increment seconds.
    parsed = df["time_control"].apply(parse_time_control)
    df["base_time"] = parsed.apply(lambda t: t[0])
    df["increment"] = parsed.apply(lambda t: t[1])

    # Encode time_class and build the remaining Option-A features.
    df["time_class_encoded"] = (
        df["time_class"].str.lower().map(TIME_CLASS_MAP).fillna(1).astype(int)
    )
    df["rating_diff"] = df["white_rating"] - df["black_rating"]
    df["abs_rating_diff"] = df["rating_diff"].abs()
    df["rated"] = df["rated"].astype(bool).astype(int)

    # Assemble the final processed table. base_time/increment are no longer
    # model features but are kept in the CSV for future use.
    output_columns = FeatureEngineer.FEATURE_NAMES + [
        "base_time",
        "increment",
        "target",
        "termination_reason",
    ]
    processed = df[output_columns].copy()

    # Report the class distribution.
    print("Class distribution (target):")
    print(processed["target"].value_counts().sort_index())
    print(f"Usable rows: {len(processed)}")

    # Resolve output paths relative to the project root.
    processed_path = os.path.join(PROJECT_ROOT, save_path)
    raw_out_path = os.path.join(PROJECT_ROOT, "data", "raw", "games_raw.csv")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    os.makedirs(os.path.dirname(raw_out_path), exist_ok=True)

    processed.to_csv(processed_path, index=False)
    df.to_csv(raw_out_path, index=False)
    print(f"Processed dataset saved to {processed_path}")
    print(f"Filtered raw frame saved to {raw_out_path}")

    return processed


if __name__ == "__main__":
    build_processed_dataset()
