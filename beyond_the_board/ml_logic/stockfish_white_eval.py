"""
STOCKFISH EVALUATION FOR WHITE POSITIONS (BEFORE BLACK MOVES)
Evaluates positions after White's move, before Black responds.
Uses fen_*_white columns to compute stockfish_*_white scores.
"""

import pandas as pd
import numpy as np
import chess
import chess.engine
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

STOCKFISH_PATHS = [
    "/opt/homebrew/bin/stockfish",
    "/usr/local/bin/stockfish",
    "/usr/bin/stockfish",
    "C:\\Program Files\\stockfish\\stockfish.exe",
    "stockfish",
]

EVAL_DEPTH = 10
EVAL_TIME_LIMIT = 0.02
BATCH_SIZE = 2000

# Column names for the 10 evaluation points
EVAL_POINT_NAMES = [
    'opening',
    'early',
    'developing',
    'pre_mid',
    'midgame',
    'post_mid',
    'transition',
    'late',
    'pre_end',
    'endgame'
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_stockfish():
    """Find Stockfish installation"""
    for path in STOCKFISH_PATHS:
        if os.path.exists(path):
            return path
    return None

def centipawns_to_score(cp_score):
    """Convert centipawn score to readable format (-10 to +10)"""
    if cp_score is None:
        return 0.0
    score = max(-1000, min(1000, cp_score)) / 100.0
    return round(score, 2)

def evaluate_position(board, engine, depth=10, time_limit=0.02):
    """Evaluate a single position with Stockfish"""
    try:
        info = engine.analyse(
            board,
            chess.engine.Limit(depth=depth, time=time_limit),
            options={"Threads": 1}
        )

        score = info["score"].white()

        if score.is_mate():
            mate_in = score.mate()
            return 10.0 if mate_in > 0 else -10.0
        else:
            cp = score.score()
            return centipawns_to_score(cp)

    except Exception as e:
        return 0.0

def evaluate_single_game_white(args):
    """
    Evaluate WHITE positions (fen_*_white) for a single game

    Args:
        args: Tuple of (game_index, fen_dict, stockfish_path)
        fen_dict contains: {
            'fen_opening_white': ...,
            'fen_early_white': ...,
            ...
        }

    Returns:
        Tuple of (game_index, eval1, eval2, ..., eval10)
    """
    idx, fen_dict, stockfish_path = args

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

        evaluations = []

        # Evaluate each of the 10 white positions
        for name in EVAL_POINT_NAMES:
            fen_col = f'fen_{name}_white'
            fen = fen_dict.get(fen_col)

            # Check if FEN is valid
            if pd.isna(fen) or str(fen).strip() in ['', 'nan', 'None']:
                evaluations.append(0.0)
                continue

            try:
                board = chess.Board(fen)
                eval_score = evaluate_position(board, engine, EVAL_DEPTH, EVAL_TIME_LIMIT)
                evaluations.append(eval_score)
            except:
                evaluations.append(0.0)

        engine.quit()
        return (idx,) + tuple(evaluations)

    except Exception as e:
        try:
            engine.quit()
        except:
            pass
        return (idx,) + (0.0,) * 10

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def add_stockfish_white_evaluations(
    input_csv,
    output_csv,
    stockfish_path=None,
    num_workers=None,
    batch_size=None,
    max_games=None
):
    """
    Add Stockfish evaluations for WHITE positions (before Black moves)

    Creates 10 new columns:
    - stockfish_opening_white
    - stockfish_early_white
    - stockfish_developing_white
    - stockfish_pre_mid_white
    - stockfish_midgame_white
    - stockfish_post_mid_white
    - stockfish_transition_white
    - stockfish_late_white
    - stockfish_pre_end_white
    - stockfish_endgame_white
    """

    print(f"\n{'='*70}")
    print("STOCKFISH EVALUATION FOR WHITE POSITIONS")
    print(f"{'='*70}\n")

    # Find Stockfish
    if stockfish_path is None:
        stockfish_path = find_stockfish()

    if stockfish_path is None:
        print("❌ ERROR: Stockfish not found!")
        return None

    print(f"✓ Found Stockfish: {stockfish_path}")

    # Test Stockfish
    print("✓ Testing Stockfish...")
    try:
        test_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        test_engine.quit()
        print("✓ Stockfish working!\n")
    except Exception as e:
        print(f"❌ Stockfish test failed: {e}")
        return None

    # Load data
    print(f"📂 Loading: {input_csv}")
    df = pd.read_csv(input_csv)
    original_rows = len(df)
    print(f"✓ Loaded {original_rows:,} games\n")

    # Limit games if specified
    if max_games and max_games < len(df):
        df = df.head(max_games).copy()
        print(f"⚠️  Limited to first {max_games:,} games for testing\n")

    # Auto-detect CPU cores
    if num_workers is None:
        num_workers = cpu_count()

    if batch_size is None:
        batch_size = BATCH_SIZE

    print(f"{'='*70}")
    print("CONFIGURATION")
    print(f"{'='*70}")
    print(f"Games to process:     {len(df):,}")
    print(f"Parallel workers:     {num_workers}")
    print(f"Batch size:           {batch_size:,}")
    print(f"Search depth:         {EVAL_DEPTH}")
    print(f"Time per position:    {EVAL_TIME_LIMIT}s")
    print(f"Positions per game:   10 (white positions)\n")

    estimated_throughput = num_workers * 2.7
    est_time_total = len(df) / estimated_throughput / 60
    print(f"Estimated time:       ~{est_time_total:.1f} minutes (~{est_time_total/60:.1f} hours)")
    print(f"{'='*70}\n")

    # Initialize columns for white evaluations
    for name in EVAL_POINT_NAMES:
        df[f'stockfish_{name}_white'] = 0.0

    # Process in batches
    num_batches = (len(df) + batch_size - 1) // batch_size

    print(f"🚀 Starting parallel evaluation ({num_batches} batches)...\n")

    start_time = time.time()
    total_processed = 0

    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, len(df))

        print(f"[Batch {batch_num + 1}/{num_batches}] Processing games {batch_start:,} to {batch_end:,}...")

        # Prepare arguments - extract FEN columns for white positions
        batch_args = []
        for i in range(batch_start, batch_end):
            fen_dict = {}
            for name in EVAL_POINT_NAMES:
                fen_col = f'fen_{name}_white'
                fen_dict[fen_col] = df.iloc[i][fen_col] if fen_col in df.columns else None
            batch_args.append((i, fen_dict, stockfish_path))

        # Parallel evaluation
        with Pool(processes=num_workers) as pool:
            results = pool.map(evaluate_single_game_white, batch_args)

        # Update dataframe with evaluations
        for result in results:
            idx = result[0]
            evals = result[1:]
            for i, name in enumerate(EVAL_POINT_NAMES):
                df.at[idx, f'stockfish_{name}_white'] = evals[i]

        total_processed += len(results)
        elapsed = time.time() - start_time
        games_per_sec = total_processed / elapsed if elapsed > 0 else 0
        remaining = (len(df) - total_processed) / games_per_sec if games_per_sec > 0 else 0

        print(f"  ✓ Batch {batch_num + 1} complete!")
        print(f"  Progress: {total_processed:,}/{len(df):,} ({100*total_processed/len(df):.1f}%)")
        print(f"  Speed: {games_per_sec:.1f} games/sec")
        print(f"  Remaining: ~{remaining/60:.1f} minutes\n")

        # Save progress after each batch
        df.to_csv(output_csv, index=False)

    total_time = time.time() - start_time

    print(f"{'='*70}")
    print("✅ EVALUATION COMPLETE!")
    print(f"{'='*70}\n")
    print(f"Total time:           {total_time/60:.1f} minutes")
    print(f"Games processed:      {total_processed:,}")
    print(f"Average speed:        {total_processed/total_time:.1f} games/sec\n")

    # Save final
    print(f"💾 Saving to: {output_csv}")
    df.to_csv(output_csv, index=False)

    print(f"\n{'='*70}")
    print("✅ SUCCESS!")
    print(f"{'='*70}\n")
    print("10 new WHITE evaluation columns added:")
    for name in EVAL_POINT_NAMES:
        print(f"  • stockfish_{name}_white")
    print()

    return df
