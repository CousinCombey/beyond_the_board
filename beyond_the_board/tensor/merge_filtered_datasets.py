"""
Merge filtered White and Black datasets into a unified, clean dataset

This script takes the two filtered datasets (white-to-move and black-to-move)
and merges them into a single well-structured dataset where each position
has only one FEN and one Stockfish evaluation.

The merged dataset will have:
- Columns for each stage: fen_stage, stockfish_stage, player_stage
- All other game metadata preserved
- No NaN values in the position columns (clean structure)

Usage:
    python merge_filtered_datasets.py

Or import:
    from merge_filtered_datasets import merge_filtered_datasets
    df_merged = merge_filtered_datasets('white.csv', 'black.csv')
"""

import pandas as pd
import numpy as np
from pathlib import Path


def merge_filtered_datasets(white_csv, black_csv, output_csv=None):
    """
    Merge white-to-move and black-to-move filtered datasets into one clean dataset.

    Args:
        white_csv: Path to white-to-move dataset
        black_csv: Path to black-to-move dataset
        output_csv: Optional path to save merged dataset

    Returns:
        DataFrame with merged, clean structure
    """
    print(f"\n{'='*70}")
    print("MERGING WHITE AND BLACK FILTERED DATASETS")
    print(f"{'='*70}\n")

    # Load datasets
    print(f"📂 Loading white dataset: {white_csv}")
    df_white = pd.read_csv(white_csv)
    print(f"✓ Loaded {len(df_white):,} games\n")

    print(f"📂 Loading black dataset: {black_csv}")
    df_black = pd.read_csv(black_csv)
    print(f"✓ Loaded {len(df_black):,} games\n")

    # Verify they have the same number of games
    if len(df_white) != len(df_black):
        print(f"⚠️  WARNING: Datasets have different lengths!")
        print(f"   White: {len(df_white):,} games")
        print(f"   Black: {len(df_black):,} games")
        print(f"   Will use minimum length for merging\n")

    # Start with base dataset (non-position columns)
    stages = ["opening", "early", "developing", "pre_mid", "midgame",
              "post_mid", "transition", "late", "pre_end", "endgame"]

    # Identify columns to exclude from base (position-specific columns)
    position_cols = []
    for stage in stages:
        position_cols.extend([
            f"fen_{stage}_white",
            f"fen_{stage}_black",
            f"stockfish_{stage}_white",
            f"stockfish_{stage}_black",
            f"move_{stage}_white",
            f"move_{stage}_black"
        ])

    # Get base columns (metadata that's the same in both datasets)
    base_cols = [col for col in df_white.columns if col not in position_cols]

    print(f"📊 Dataset structure:")
    print(f"   Base metadata columns: {len(base_cols)}")
    print(f"   Position stages: {len(stages)}")
    print(f"   Total positions to merge: {len(stages) * 2} (white + black)\n")

    # Start with base dataset from white (could be either)
    df_merged = df_white[base_cols].copy()

    print(f"🔄 Merging position data...\n")

    # For each stage, merge the white and black positions
    for stage in stages:
        white_fen_col = f"fen_{stage}_white"
        black_fen_col = f"fen_{stage}_black"
        white_sf_col = f"stockfish_{stage}_white"
        black_sf_col = f"stockfish_{stage}_black"
        white_move_col = f"move_{stage}_white"
        black_move_col = f"move_{stage}_black"

        # Create unified columns for this stage
        merged_fen_col = f"fen_{stage}"
        merged_sf_col = f"stockfish_{stage}"
        merged_move_col = f"move_{stage}"
        merged_player_col = f"player_{stage}"

        # Initialize merged columns
        df_merged[merged_fen_col] = None
        df_merged[merged_sf_col] = None
        df_merged[merged_move_col] = None
        df_merged[merged_player_col] = None

        # Merge white positions from white dataset
        if white_fen_col in df_white.columns:
            white_mask = df_white[white_fen_col].notna()
            df_merged.loc[white_mask, merged_fen_col] = df_white.loc[white_mask, white_fen_col]
            if white_sf_col in df_white.columns:
                df_merged.loc[white_mask, merged_sf_col] = df_white.loc[white_mask, white_sf_col]
            if white_move_col in df_white.columns:
                df_merged.loc[white_mask, merged_move_col] = df_white.loc[white_mask, white_move_col]
            df_merged.loc[white_mask, merged_player_col] = 'w'

        # Merge black positions from white dataset
        if black_fen_col in df_white.columns:
            black_mask = df_white[black_fen_col].notna()
            df_merged.loc[black_mask, merged_fen_col] = df_white.loc[black_mask, black_fen_col]
            if black_sf_col in df_white.columns:
                df_merged.loc[black_mask, merged_sf_col] = df_white.loc[black_mask, black_sf_col]
            if black_move_col in df_white.columns:
                df_merged.loc[black_mask, merged_move_col] = df_white.loc[black_mask, black_move_col]
            df_merged.loc[black_mask, merged_player_col] = 'b'

        # Fill in any missing positions from black dataset
        missing_mask = df_merged[merged_fen_col].isna()

        if white_fen_col in df_black.columns:
            white_avail = missing_mask & df_black[white_fen_col].notna()
            if white_avail.any():
                df_merged.loc[white_avail, merged_fen_col] = df_black.loc[white_avail, white_fen_col]
                if white_sf_col in df_black.columns:
                    df_merged.loc[white_avail, merged_sf_col] = df_black.loc[white_avail, white_sf_col]
                if white_move_col in df_black.columns:
                    df_merged.loc[white_avail, merged_move_col] = df_black.loc[white_avail, white_move_col]
                df_merged.loc[white_avail, merged_player_col] = 'w'

        if black_fen_col in df_black.columns:
            black_avail = missing_mask & df_black[black_fen_col].notna()
            if black_avail.any():
                df_merged.loc[black_avail, merged_fen_col] = df_black.loc[black_avail, black_fen_col]
                if black_sf_col in df_black.columns:
                    df_merged.loc[black_avail, merged_sf_col] = df_black.loc[black_avail, black_sf_col]
                if black_move_col in df_black.columns:
                    df_merged.loc[black_avail, merged_move_col] = df_black.loc[black_avail, black_move_col]
                df_merged.loc[black_avail, merged_player_col] = 'b'

    # Print statistics
    print(f"{'='*70}")
    print("MERGE STATISTICS")
    print(f"{'='*70}\n")

    for stage in stages:
        fen_col = f"fen_{stage}"
        sf_col = f"stockfish_{stage}"
        player_col = f"player_{stage}"

        total = df_merged[fen_col].notna().sum()
        white_count = (df_merged[player_col] == 'w').sum()
        black_count = (df_merged[player_col] == 'b').sum()

        print(f"{stage:15s}: {total:6,} positions (W: {white_count:6,}, B: {black_count:6,})")

    total_positions = sum(df_merged[f"fen_{stage}"].notna().sum() for stage in stages)
    print(f"\n{'Total positions':15s}: {total_positions:,}")
    print(f"{'Games':15s}: {len(df_merged):,}\n")

    # Save if output path provided
    if output_csv:
        print(f"💾 Saving merged dataset to: {output_csv}")
        df_merged.to_csv(output_csv, index=False)
        print(f"✓ Saved {len(df_merged):,} games\n")

    print(f"{'='*70}")
    print("✅ MERGE COMPLETE!")
    print(f"{'='*70}\n")

    print("New dataset structure:")
    print(f"  For each stage (e.g., 'opening'):")
    print(f"    • fen_opening: The position (FEN string)")
    print(f"    • stockfish_opening: The evaluation")
    print(f"    • player_opening: Who is to move ('w' or 'b')")
    print(f"    • move_opening: The move played (if available)\n")

    return df_merged


def create_stage_specific_datasets(merged_csv, output_dir="data/stages"):
    """
    Optional: Split merged dataset into stage-specific datasets.

    Creates separate CSV files for each game stage, with only the positions
    from that stage.

    Args:
        merged_csv: Path to merged dataset
        output_dir: Directory to save stage-specific datasets

    Returns:
        Dictionary of {stage: dataframe}
    """
    print(f"\n{'='*70}")
    print("CREATING STAGE-SPECIFIC DATASETS")
    print(f"{'='*70}\n")

    # Load merged dataset
    print(f"📂 Loading merged dataset: {merged_csv}")
    df = pd.read_csv(merged_csv)
    print(f"✓ Loaded {len(df):,} games\n")

    stages = ["opening", "early", "developing", "pre_mid", "midgame",
              "post_mid", "transition", "late", "pre_end", "endgame"]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stage_dfs = {}

    for stage in stages:
        fen_col = f"fen_{stage}"
        sf_col = f"stockfish_{stage}"
        player_col = f"player_{stage}"
        move_col = f"move_{stage}"

        # Create dataframe for this stage
        stage_df = pd.DataFrame()

        # Add metadata columns (non-position columns)
        metadata_cols = [col for col in df.columns
                        if not any(s in col for s in ['fen_', 'stockfish_', 'player_', 'move_'])]

        for col in metadata_cols:
            stage_df[col] = df[col]

        # Add position columns (renamed without stage suffix)
        if fen_col in df.columns:
            stage_df['fen'] = df[fen_col]
        if sf_col in df.columns:
            stage_df['stockfish'] = df[sf_col]
        if player_col in df.columns:
            stage_df['player'] = df[player_col]
        if move_col in df.columns:
            stage_df['move'] = df[move_col]

        # Remove rows where position is missing
        stage_df = stage_df[stage_df['fen'].notna()].reset_index(drop=True)

        # Save
        output_file = output_path / f"games_{stage}.csv"
        stage_df.to_csv(output_file, index=False)

        stage_dfs[stage] = stage_df

        print(f"✓ {stage:15s}: {len(stage_df):6,} positions → {output_file}")

    print(f"\n{'='*70}")
    print("✅ STAGE DATASETS CREATED!")
    print(f"{'='*70}\n")

    return stage_dfs


if __name__ == "__main__":
    """
    Example usage when running as a script
    """

    # Define paths
    white_csv = "data/games_white_to_move.csv"
    black_csv = "data/games_black_to_move.csv"
    output_merged = "data/games_merged_clean.csv"

    # Check if filtered datasets exist
    if not Path(white_csv).exists() or not Path(black_csv).exists():
        print("❌ ERROR: Filtered datasets not found!")
        print(f"   Expected: {white_csv}")
        print(f"   Expected: {black_csv}")
        print("\n   Run filter_by_turn.py first to create filtered datasets.")
        exit(1)

    # Merge datasets
    df_merged = merge_filtered_datasets(
        white_csv=white_csv,
        black_csv=black_csv,
        output_csv=output_merged
    )

    # Optional: Create stage-specific datasets
    print("\n" + "="*70)
    user_input = input("Create stage-specific datasets? (y/n): ")
    if user_input.lower() == 'y':
        stage_dfs = create_stage_specific_datasets(output_merged)

    print("\n✅ All done!")
    print(f"\nMerged dataset: {output_merged}")
