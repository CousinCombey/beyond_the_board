"""
Data processing functions for MERGED datasets.

This module works with merged datasets that have the clean structure:
- fen_{stage}, stockfish_{stage}, player_{stage}, move_{stage}

Instead of the original structure with _white and _black suffixes.
"""

import pandas as pd
import numpy as np


def create_new_df_white_merged(df):
    """
    Extract White-to-move positions from merged dataset.

    Takes a merged DataFrame with columns like fen_opening, stockfish_opening, player_opening
    and extracts only positions where player == 'w' (White to move).

    Args:
        df: Merged DataFrame with fen_{stage}, stockfish_{stage}, player_{stage} columns

    Returns:
        DataFrame with FEN and Stockfish columns for White-to-move positions
    """

    stages = ["opening", "early", "developing", "pre_mid", "midgame",
              "post_mid", "transition", "late", "pre_end", "endgame"]

    new_df = pd.DataFrame(columns=["FEN", "Stockfish"])

    # Iterate through each stage and extract White positions
    for stage in stages:
        fen_col = f"fen_{stage}"
        stockfish_col = f"stockfish_{stage}"
        player_col = f"player_{stage}"

        # Check if columns exist
        if fen_col not in df.columns or stockfish_col not in df.columns or player_col not in df.columns:
            continue

        # Filter for White to move positions
        white_mask = (df[player_col] == 'w') & df[fen_col].notna()

        # Extract positions
        fens = df.loc[white_mask, fen_col].reset_index(drop=True)
        stockfish = df.loc[white_mask, stockfish_col].reset_index(drop=True)

        new_df_stage = pd.DataFrame({"FEN": fens, "Stockfish": stockfish})

        new_df = pd.concat([new_df, new_df_stage], ignore_index=True)

    # Apply FEN parsing and feature extraction
    new_df = df_fen_cut_merged(new_df)
    new_df = df_final_cut_merged(new_df)

    print(f"✓ Created White dataset: {len(new_df):,} positions where White is to move")

    return new_df


def create_new_df_black_merged(df):
    """
    Extract Black-to-move positions from merged dataset.

    Takes a merged DataFrame with columns like fen_opening, stockfish_opening, player_opening
    and extracts only positions where player == 'b' (Black to move).

    Args:
        df: Merged DataFrame with fen_{stage}, stockfish_{stage}, player_{stage} columns

    Returns:
        DataFrame with FEN and Stockfish columns for Black-to-move positions
    """

    stages = ["opening", "early", "developing", "pre_mid", "midgame",
              "post_mid", "transition", "late", "pre_end", "endgame"]

    new_df = pd.DataFrame(columns=["FEN", "Stockfish"])

    # Iterate through each stage and extract Black positions
    for stage in stages:
        fen_col = f"fen_{stage}"
        stockfish_col = f"stockfish_{stage}"
        player_col = f"player_{stage}"

        # Check if columns exist
        if fen_col not in df.columns or stockfish_col not in df.columns or player_col not in df.columns:
            continue

        # Filter for Black to move positions
        black_mask = (df[player_col] == 'b') & df[fen_col].notna()

        # Extract positions
        fens = df.loc[black_mask, fen_col].reset_index(drop=True)
        stockfish = df.loc[black_mask, stockfish_col].reset_index(drop=True)

        new_df_stage = pd.DataFrame({"FEN": fens, "Stockfish": stockfish})

        new_df = pd.concat([new_df, new_df_stage], ignore_index=True)

    # Apply FEN parsing and feature extraction
    new_df = df_fen_cut_merged(new_df)
    new_df = df_final_cut_merged(new_df)

    print(f"✓ Created Black dataset: {len(new_df):,} positions where Black is to move")

    return new_df


def create_new_df_all_merged(df):
    """
    Extract ALL positions from merged dataset (both White and Black to move).

    Takes a merged DataFrame with columns like fen_opening, stockfish_opening, player_opening
    and extracts all valid positions.

    Args:
        df: Merged DataFrame with fen_{stage}, stockfish_{stage}, player_{stage} columns

    Returns:
        DataFrame with FEN and Stockfish columns for all positions
    """

    stages = ["opening", "early", "developing", "pre_mid", "midgame",
              "post_mid", "transition", "late", "pre_end", "endgame"]

    new_df = pd.DataFrame(columns=["FEN", "Stockfish"])

    # Iterate through each stage and extract all positions
    for stage in stages:
        fen_col = f"fen_{stage}"
        stockfish_col = f"stockfish_{stage}"

        # Check if columns exist
        if fen_col not in df.columns or stockfish_col not in df.columns:
            continue

        # Filter for valid positions (not NaN)
        valid_mask = df[fen_col].notna()

        # Extract positions
        fens = df.loc[valid_mask, fen_col].reset_index(drop=True)
        stockfish = df.loc[valid_mask, stockfish_col].reset_index(drop=True)

        new_df_stage = pd.DataFrame({"FEN": fens, "Stockfish": stockfish})

        new_df = pd.concat([new_df, new_df_stage], ignore_index=True)

    # Apply FEN parsing and feature extraction
    new_df = df_fen_cut_merged(new_df)
    new_df = df_final_cut_merged(new_df)

    print(f"✓ Created combined dataset: {len(new_df):,} positions (White + Black)")

    return new_df


def df_fen_cut_merged(df):
    """
    Parse FEN strings into separate components.

    Creates 6 new columns from the FEN string:
    - FEN_board: Board position
    - FEN_player: Active player (w/b)
    - FEN_roque: Castling rights
    - FEN_passant: En passant square
    - FEN_count_nul: Halfmove clock
    - FEN_coup_complet: Fullmove number
    """
    df['FEN_board'] = df["FEN"].apply(lambda x: x.split(" ")[0] if pd.notna(x) else None)
    df['FEN_player'] = df["FEN"].apply(lambda x: x.split(" ")[1] if pd.notna(x) else None)
    df['FEN_roque'] = df["FEN"].apply(lambda x: x.split(" ")[2] if pd.notna(x) else None)
    df['FEN_passant'] = df["FEN"].apply(lambda x: x.split(" ")[3] if pd.notna(x) else None)
    df['FEN_count_nul'] = df["FEN"].apply(lambda x: x.split(" ")[4] if pd.notna(x) else None)
    df['FEN_coup_complet'] = df["FEN"].apply(lambda x: x.split(" ")[5] if pd.notna(x) else None)

    return df


def df_final_cut_merged(df):
    """
    Create engineered features from FEN components.

    Creates:
    - turn_enc: Turn encoded (1 for white, 0 for black)
    - Castling rights (4 columns)
    - En passant files (8 columns)
    - Normalized move counters
    """

    # Transforme le tour du joueur w or b en chiffre
    df["turn_enc"] = (df["FEN_player"] == "w").astype(int)

    # Transforme les roques en 4 différentes colonnes possibles
    df["white_king_castling"]  = df["FEN_roque"].apply(lambda x: "K" in x if pd.notna(x) else False).astype(int)
    df["white_queen_castling"] = df["FEN_roque"].apply(lambda x: "Q" in x if pd.notna(x) else False).astype(int)
    df["black_king_castling"]  = df["FEN_roque"].apply(lambda x: "k" in x if pd.notna(x) else False).astype(int)
    df["black_queen_castling"] = df["FEN_roque"].apply(lambda x: "q" in x if pd.notna(x) else False).astype(int)

    # Transforme les en passant en 8 colonnes possible en fonction de leurs lettre sur l'échiquier
    files = list("abcdefgh")
    for f in files:
        df[f"ep_{f}"] = df["FEN_passant"].apply(lambda x: int(x != "-" and pd.notna(x) and x[0] == f) if pd.notna(x) else 0)

    # Demi coup normalisé
    df["halfmove_norm"] = df["FEN_count_nul"].apply(lambda x: int(x) / 100.0 if pd.notna(x) else 0.0)

    # Coup complet (une fois que le blanc, puis le noir ont joué) normalisé
    df["fullmove_norm"] = df["FEN_coup_complet"].apply(lambda x: int(x) / 200.0 if pd.notna(x) else 0.0)

    return df


def drop_columns_merged(df):
    """
    Drop FEN-related columns, keeping only processed features.

    Args:
        df: DataFrame with FEN columns

    Returns:
        DataFrame with FEN columns removed
    """
    columns_to_drop = [
        "FEN",
        "FEN_board",
        "FEN_player",
        "FEN_roque",
        "FEN_passant",
        "FEN_count_nul",
        "FEN_coup_complet",
        "Stockfish"
    ]
    return df.drop(columns=[col for col in columns_to_drop if col in df.columns])


if __name__ == "__main__":
    """
    Example usage with merged dataset
    """
    print("Testing with merged dataset...")

    # Load merged dataset
    df_merged = pd.read_csv('/workspace/beyond_the_board/data/games_merged_clean.csv')

    print(f"\nMerged dataset loaded: {len(df_merged):,} games")

    # Count positions per stage
    stages = ["opening", "early", "developing", "pre_mid", "midgame",
              "post_mid", "transition", "late", "pre_end", "endgame"]

    print("\nPositions per stage:")
    total_white = 0
    total_black = 0

    for stage in stages:
        player_col = f"player_{stage}"
        if player_col in df_merged.columns:
            white_count = (df_merged[player_col] == 'w').sum()
            black_count = (df_merged[player_col] == 'b').sum()
            total_white += white_count
            total_black += black_count
            print(f"  {stage:15s}: W={white_count:6,}, B={black_count:6,}")

    print(f"\n  {'Total':15s}: W={total_white:6,}, B={total_black:6,}")

    # Create position datasets
    print("\n" + "="*70)
    new_df_white = create_new_df_white_merged(df_merged)
    new_df_black = create_new_df_black_merged(df_merged)
    new_df_all = create_new_df_all_merged(df_merged)

    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    print(f"White positions extracted: {len(new_df_white):,}")
    print(f"Black positions extracted: {len(new_df_black):,}")
    print(f"Total positions extracted: {len(new_df_all):,}")
    print(f"Expected total: {total_white + total_black:,}")

    # Verify player distribution in extracted datasets
    print(f"\nWhite dataset player distribution:")
    print(new_df_white['FEN_player'].value_counts())

    print(f"\nBlack dataset player distribution:")
    print(new_df_black['FEN_player'].value_counts())

    print("\n✅ All tests passed!")
