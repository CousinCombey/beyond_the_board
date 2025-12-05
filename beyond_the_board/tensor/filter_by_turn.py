"""
Filter chess dataset by player turn (White to move vs Black to move)

This script takes the original dataset with FEN positions and creates two filtered datasets:
1. Dataset with only positions where White is to move
2. Dataset with only positions where Black is to move

Usage:
    python filter_by_turn.py

Or import and use:
    from filter_by_turn import filter_dataset_by_turn
    df_white, df_black = filter_dataset_by_turn('input.csv')
"""

import pandas as pd
import numpy as np
from pathlib import Path


def extract_player_from_fen(fen_string):
    """
    Extract the active player from a FEN string.

    Args:
        fen_string: Full FEN notation (e.g., "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    Returns:
        'w' for white to move, 'b' for black to move, or None if invalid
    """
    if pd.isna(fen_string) or str(fen_string).strip() in ['', 'nan', 'None']:
        return None

    try:
        parts = str(fen_string).split()
        if len(parts) >= 2:
            return parts[1]  # Second field is the active player
    except:
        return None

    return None


def filter_white_positions(df):
    """
    Filter dataset to only include positions where White is to move.

    Checks all FEN columns (fen_*_white and fen_*_black) and only keeps rows
    where the position is White's turn. Also filters corresponding Stockfish columns.

    Args:
        df: Original dataframe with FEN columns

    Returns:
        DataFrame with only white-to-move positions
    """
    stages = ["opening", "early", "developing", "pre_mid", "midgame",
              "post_mid", "transition", "late", "pre_end", "endgame"]

    df_filtered = df.copy()

    print("Filtering White positions (White to move)...")

    # For each stage, check both _white and _black FEN columns
    for stage in stages:
        white_fen_col = f"fen_{stage}_white"
        black_fen_col = f"fen_{stage}_black"
        white_sf_col = f"stockfish_{stage}_white"
        black_sf_col = f"stockfish_{stage}_black"

        # Filter white FEN column and its corresponding Stockfish
        if white_fen_col in df_filtered.columns:
            # Extract player turn from FEN
            player_turn = df_filtered[white_fen_col].apply(extract_player_from_fen)
            # Keep only positions where it's white's turn
            mask = player_turn != 'w'
            df_filtered.loc[mask, white_fen_col] = np.nan
            # Also filter the corresponding Stockfish score
            if white_sf_col in df_filtered.columns:
                df_filtered.loc[mask, white_sf_col] = np.nan

        # Filter black FEN column and its corresponding Stockfish
        if black_fen_col in df_filtered.columns:
            player_turn = df_filtered[black_fen_col].apply(extract_player_from_fen)
            # Keep only positions where it's white's turn
            mask = player_turn != 'w'
            df_filtered.loc[mask, black_fen_col] = np.nan
            # Also filter the corresponding Stockfish score
            if black_sf_col in df_filtered.columns:
                df_filtered.loc[mask, black_sf_col] = np.nan

    return df_filtered


def filter_black_positions(df):
    """
    Filter dataset to only include positions where Black is to move.

    Checks all FEN columns (fen_*_white and fen_*_black) and only keeps rows
    where the position is Black's turn. Also filters corresponding Stockfish columns.

    Args:
        df: Original dataframe with FEN columns

    Returns:
        DataFrame with only black-to-move positions
    """
    stages = ["opening", "early", "developing", "pre_mid", "midgame",
              "post_mid", "transition", "late", "pre_end", "endgame"]

    df_filtered = df.copy()

    print("Filtering Black positions (Black to move)...")

    # For each stage, check both _white and _black FEN columns
    for stage in stages:
        white_fen_col = f"fen_{stage}_white"
        black_fen_col = f"fen_{stage}_black"
        white_sf_col = f"stockfish_{stage}_white"
        black_sf_col = f"stockfish_{stage}_black"

        # Filter white FEN column and its corresponding Stockfish
        if white_fen_col in df_filtered.columns:
            # Extract player turn from FEN
            player_turn = df_filtered[white_fen_col].apply(extract_player_from_fen)
            # Keep only positions where it's black's turn
            mask = player_turn != 'b'
            df_filtered.loc[mask, white_fen_col] = np.nan
            # Also filter the corresponding Stockfish score
            if white_sf_col in df_filtered.columns:
                df_filtered.loc[mask, white_sf_col] = np.nan

        # Filter black FEN column and its corresponding Stockfish
        if black_fen_col in df_filtered.columns:
            player_turn = df_filtered[black_fen_col].apply(extract_player_from_fen)
            # Keep only positions where it's black's turn
            mask = player_turn != 'b'
            df_filtered.loc[mask, black_fen_col] = np.nan
            # Also filter the corresponding Stockfish score
            if black_sf_col in df_filtered.columns:
                df_filtered.loc[mask, black_sf_col] = np.nan

    return df_filtered


def filter_dataset_by_turn(input_csv, output_white_csv=None, output_black_csv=None):
    """
    Main function to filter dataset by player turn.

    Args:
        input_csv: Path to input CSV file
        output_white_csv: Path to save white-to-move dataset (optional)
        output_black_csv: Path to save black-to-move dataset (optional)

    Returns:
        Tuple of (df_white, df_black) dataframes
    """
    print(f"\n{'='*70}")
    print("FILTERING CHESS DATASET BY PLAYER TURN")
    print(f"{'='*70}\n")

    # Load data
    print(f"📂 Loading: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"✓ Loaded {len(df):,} games\n")

    # Count original FEN positions
    stages = ["opening", "early", "developing", "pre_mid", "midgame",
              "post_mid", "transition", "late", "pre_end", "endgame"]

    original_white_fens = 0
    original_black_fens = 0

    for stage in stages:
        white_col = f"fen_{stage}_white"
        black_col = f"fen_{stage}_black"
        if white_col in df.columns:
            original_white_fens += df[white_col].notna().sum()
        if black_col in df.columns:
            original_black_fens += df[black_col].notna().sum()

    print(f"Original FEN positions in dataset:")
    print(f"  fen_*_white columns: {original_white_fens:,}")
    print(f"  fen_*_black columns: {original_black_fens:,}")
    print(f"  Total: {original_white_fens + original_black_fens:,}\n")

    # Filter for white positions
    print(f"{'='*70}")
    df_white = filter_white_positions(df)

    # Count filtered white FENs
    filtered_white_fens = 0
    for stage in stages:
        white_col = f"fen_{stage}_white"
        black_col = f"fen_{stage}_black"
        if white_col in df_white.columns:
            filtered_white_fens += df_white[white_col].notna().sum()
        if black_col in df_white.columns:
            filtered_white_fens += df_white[black_col].notna().sum()

    print(f"✓ White dataset: {filtered_white_fens:,} positions where White is to move")
    print(f"  (Filtered out {original_white_fens + original_black_fens - filtered_white_fens:,} positions)\n")

    # Filter for black positions
    print(f"{'='*70}")
    df_black = filter_black_positions(df)

    # Count filtered black FENs
    filtered_black_fens = 0
    for stage in stages:
        white_col = f"fen_{stage}_white"
        black_col = f"fen_{stage}_black"
        if white_col in df_black.columns:
            filtered_black_fens += df_black[white_col].notna().sum()
        if black_col in df_black.columns:
            filtered_black_fens += df_black[black_col].notna().sum()

    print(f"✓ Black dataset: {filtered_black_fens:,} positions where Black is to move")
    print(f"  (Filtered out {original_white_fens + original_black_fens - filtered_black_fens:,} positions)\n")

    # Save if output paths provided
    if output_white_csv:
        print(f"💾 Saving White dataset to: {output_white_csv}")
        df_white.to_csv(output_white_csv, index=False)
        print(f"✓ Saved {len(df_white):,} games\n")

    if output_black_csv:
        print(f"💾 Saving Black dataset to: {output_black_csv}")
        df_black.to_csv(output_black_csv, index=False)
        print(f"✓ Saved {len(df_black):,} games\n")

    print(f"{'='*70}")
    print("✅ FILTERING COMPLETE!")
    print(f"{'='*70}\n")

    return df_white, df_black


if __name__ == "__main__":
    """
    Example usage when running as a script
    """
    from beyond_the_board.params import FINAL_DATASET

    # Define output paths
    input_path = FINAL_DATASET
    output_white = "data/games_white_to_move.csv"
    output_black = "data/games_black_to_move.csv"

    # Create output directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)

    # Filter dataset
    df_white, df_black = filter_dataset_by_turn(
        input_path,
        output_white_csv=output_white,
        output_black_csv=output_black
    )

    print("Done! You can now use these filtered datasets for training:")
    print(f"  White dataset: {output_white}")
    print(f"  Black dataset: {output_black}")
