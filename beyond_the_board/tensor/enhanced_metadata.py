"""
Enhanced Metadata Feature Extraction for Chess Positions

Extracts 40 tactical and strategic features from FEN strings:
- Original 15 features (turn, castling, en passant, move counters)
- 25 new tactical features (checks, material, king safety, threats)

Author: Claude Code
Created: 2025-12-08
"""

import chess
import numpy as np
import pandas as pd
from typing import Dict, Tuple


# Piece values for material calculation
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}


def extract_basic_fen_features(fen: str) -> Dict[str, float]:
    """
    Extract the original 15 features from FEN string.

    Features:
    - turn_enc (1)
    - castling rights (4)
    - en passant (8)
    - move counters (2)

    Args:
        fen: FEN string

    Returns:
        Dictionary with 15 features
    """
    parts = fen.split()

    # Turn encoding
    turn_enc = 1 if parts[1] == 'w' else 0

    # Castling rights
    castling = parts[2]
    white_king_castling = int('K' in castling)
    white_queen_castling = int('Q' in castling)
    black_king_castling = int('k' in castling)
    black_queen_castling = int('q' in castling)

    # En passant
    ep_square = parts[3]
    ep_features = {f'ep_{file}': 0 for file in 'abcdefgh'}
    if ep_square != '-':
        ep_features[f'ep_{ep_square[0]}'] = 1

    # Move counters (normalized)
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    fullmove = int(parts[5]) if len(parts) > 5 else 1
    halfmove_norm = halfmove / 100.0
    fullmove_norm = fullmove / 200.0

    features = {
        'turn_enc': turn_enc,
        'white_king_castling': white_king_castling,
        'white_queen_castling': white_queen_castling,
        'black_king_castling': black_king_castling,
        'black_queen_castling': black_queen_castling,
        **ep_features,
        'halfmove_norm': halfmove_norm,
        'fullmove_norm': fullmove_norm
    }

    return features


def extract_tactical_features(board: chess.Board) -> Dict[str, float]:
    """
    Extract 8 tactical indicator features.

    Features:
    - is_in_check
    - num_checkers
    - has_escape_squares
    - can_block_check
    - can_capture_checker
    - num_legal_moves
    - is_checkmate
    - is_stalemate
    """
    is_check = board.is_check()
    is_checkmate = board.is_checkmate()
    is_stalemate = board.is_stalemate()

    # Number of checkers
    num_checkers = len(board.checkers()) if is_check else 0

    # Legal moves
    legal_moves = list(board.legal_moves)
    num_legal_moves = len(legal_moves)

    # Escape squares (king can move)
    king_square = board.king(board.turn)
    has_escape_squares = 0
    if king_square:
        king_moves = [m for m in legal_moves if m.from_square == king_square]
        has_escape_squares = int(len(king_moves) > 0)

    # Can block or capture checker
    can_block_check = 0
    can_capture_checker = 0
    if is_check and not is_checkmate:
        checkers = board.checkers()
        # Can capture any checker
        for checker_square in checkers:
            for move in legal_moves:
                if move.to_square == checker_square:
                    can_capture_checker = 1
                    break
        # Can block check (any legal move that's not king move or capture)
        if not can_capture_checker:
            for move in legal_moves:
                if move.from_square != king_square:
                    can_block_check = 1
                    break

    features = {
        'is_in_check': int(is_check),
        'num_checkers': min(num_checkers, 2) / 2.0,  # Normalize 0-2 to 0-1
        'has_escape_squares': has_escape_squares,
        'can_block_check': can_block_check,
        'can_capture_checker': can_capture_checker,
        'num_legal_moves': min(num_legal_moves, 50) / 50.0,  # Normalize to 0-1
        'is_checkmate': int(is_checkmate),
        'is_stalemate': int(is_stalemate)
    }

    return features


def extract_material_features(board: chess.Board) -> Dict[str, float]:
    """
    Extract 4 material indicator features.

    Features:
    - material_count_white
    - material_count_black
    - material_advantage
    - piece_activity
    """
    white_material = 0
    black_material = 0

    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        white_material += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
        black_material += len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]

    # Normalize material (max ~39 points per side)
    material_count_white = white_material / 39.0
    material_count_black = black_material / 39.0
    material_advantage = (white_material - black_material) / 39.0

    # Piece activity: pieces that have moved from starting squares
    starting_squares_white = {
        chess.KNIGHT: [chess.B1, chess.G1],
        chess.BISHOP: [chess.C1, chess.F1],
        chess.ROOK: [chess.A1, chess.H1],
        chess.QUEEN: [chess.D1]
    }
    starting_squares_black = {
        chess.KNIGHT: [chess.B8, chess.G8],
        chess.BISHOP: [chess.C8, chess.F8],
        chess.ROOK: [chess.A8, chess.H8],
        chess.QUEEN: [chess.D8]
    }

    developed_pieces = 0
    total_developable = 14  # 2N + 2B + 2R + 1Q per side

    for piece_type, squares in starting_squares_white.items():
        for square in squares:
            if board.piece_at(square) != chess.Piece(piece_type, chess.WHITE):
                developed_pieces += 1

    for piece_type, squares in starting_squares_black.items():
        for square in squares:
            if board.piece_at(square) != chess.Piece(piece_type, chess.BLACK):
                developed_pieces += 1

    piece_activity = developed_pieces / total_developable

    features = {
        'material_count_white': material_count_white,
        'material_count_black': material_count_black,
        'material_advantage': material_advantage,
        'piece_activity': piece_activity
    }

    return features


def extract_king_safety_features(board: chess.Board) -> Dict[str, float]:
    """
    Extract 6 king safety features.

    Features:
    - white_king_in_center
    - black_king_in_center
    - white_pawn_shield
    - black_pawn_shield
    - white_king_attackers
    - black_king_attackers
    """
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)

    # King in center (e1/d1/e2/d2 for white, e8/d8/e7/d7 for black)
    center_squares_white = [chess.E1, chess.D1, chess.E2, chess.D2]
    center_squares_black = [chess.E8, chess.D8, chess.E7, chess.D7]

    white_king_in_center = int(white_king in center_squares_white) if white_king else 0
    black_king_in_center = int(black_king in center_squares_black) if black_king else 0

    # Pawn shield (pawns near king)
    white_pawn_shield = 0
    black_pawn_shield = 0

    if white_king:
        file = chess.square_file(white_king)
        rank = chess.square_rank(white_king)
        for f in range(max(0, file-1), min(8, file+2)):
            for r in range(rank, min(8, rank+3)):
                sq = chess.square(f, r)
                piece = board.piece_at(sq)
                if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                    white_pawn_shield += 1

    if black_king:
        file = chess.square_file(black_king)
        rank = chess.square_rank(black_king)
        for f in range(max(0, file-1), min(8, file+2)):
            for r in range(max(0, rank-2), rank+1):
                sq = chess.square(f, r)
                piece = board.piece_at(sq)
                if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                    black_pawn_shield += 1

    # Normalize pawn shield (max 3)
    white_pawn_shield = min(white_pawn_shield, 3) / 3.0
    black_pawn_shield = min(black_pawn_shield, 3) / 3.0

    # King attackers (enemy pieces attacking king zone)
    white_king_attackers = 0
    black_king_attackers = 0

    if white_king:
        attackers = board.attackers(chess.BLACK, white_king)
        white_king_attackers = min(len(attackers), 5) / 5.0

    if black_king:
        attackers = board.attackers(chess.WHITE, black_king)
        black_king_attackers = min(len(attackers), 5) / 5.0

    features = {
        'white_king_in_center': white_king_in_center,
        'black_king_in_center': black_king_in_center,
        'white_pawn_shield': white_pawn_shield,
        'black_pawn_shield': black_pawn_shield,
        'white_king_attackers': white_king_attackers,
        'black_king_attackers': black_king_attackers
    }

    return features


def extract_threat_features(board: chess.Board) -> Dict[str, float]:
    """
    Extract 7 attack/threat features.

    Features:
    - num_attacked_pieces
    - num_defended_pieces
    - num_hanging_pieces
    - control_center_squares
    - control_key_files
    - bishop_pair_white
    - bishop_pair_black
    """
    turn = board.turn

    # Count attacked, defended, and hanging pieces
    attacked = 0
    defended = 0
    hanging = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == turn:
            # Is it attacked?
            is_attacked = board.is_attacked_by(not turn, square)
            if is_attacked:
                attacked += 1
                # Is it defended?
                is_defended = board.is_attacked_by(turn, square)
                if is_defended:
                    defended += 1
                else:
                    hanging += 1

    # Normalize (max 16 pieces per side)
    num_attacked_pieces = min(attacked, 16) / 16.0
    num_defended_pieces = min(defended, 16) / 16.0
    num_hanging_pieces = min(hanging, 16) / 16.0

    # Center control (e4, d4, e5, d5)
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    controlled_center = sum(1 for sq in center_squares if board.is_attacked_by(turn, sq))
    control_center_squares = controlled_center / 4.0

    # Key files control (d, e files)
    key_files = [chess.square(file, rank) for file in [3, 4] for rank in range(8)]
    controlled_files = sum(1 for sq in key_files if board.is_attacked_by(turn, sq))
    control_key_files = controlled_files / 16.0

    # Bishop pair
    white_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
    black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
    bishop_pair_white = int(white_bishops >= 2)
    bishop_pair_black = int(black_bishops >= 2)

    features = {
        'num_attacked_pieces': num_attacked_pieces,
        'num_defended_pieces': num_defended_pieces,
        'num_hanging_pieces': num_hanging_pieces,
        'control_center_squares': control_center_squares,
        'control_key_files': control_key_files,
        'bishop_pair_white': bishop_pair_white,
        'bishop_pair_black': bishop_pair_black
    }

    return features


def extract_all_features(fen: str) -> Dict[str, float]:
    """
    Extract all 40 features from a FEN string.

    Args:
        fen: FEN string

    Returns:
        Dictionary with 40 features
    """
    try:
        board = chess.Board(fen)

        # Extract all feature groups
        basic = extract_basic_fen_features(fen)
        tactical = extract_tactical_features(board)
        material = extract_material_features(board)
        king_safety = extract_king_safety_features(board)
        threats = extract_threat_features(board)

        # Combine all features
        all_features = {
            **basic,
            **tactical,
            **material,
            **king_safety,
            **threats
        }

        return all_features

    except Exception as e:
        # Return zero features if FEN is invalid
        print(f"Error processing FEN '{fen}': {e}")
        return {f'feature_{i}': 0.0 for i in range(40)}


def get_feature_names() -> list:
    """
    Get ordered list of all 40 feature names.

    Returns:
        List of feature names in correct order
    """
    return [
        # Basic features (15)
        'turn_enc',
        'white_king_castling',
        'white_queen_castling',
        'black_king_castling',
        'black_queen_castling',
        'ep_a', 'ep_b', 'ep_c', 'ep_d', 'ep_e', 'ep_f', 'ep_g', 'ep_h',
        'halfmove_norm',
        'fullmove_norm',

        # Tactical features (8)
        'is_in_check',
        'num_checkers',
        'has_escape_squares',
        'can_block_check',
        'can_capture_checker',
        'num_legal_moves',
        'is_checkmate',
        'is_stalemate',

        # Material features (4)
        'material_count_white',
        'material_count_black',
        'material_advantage',
        'piece_activity',

        # King safety features (6)
        'white_king_in_center',
        'black_king_in_center',
        'white_pawn_shield',
        'black_pawn_shield',
        'white_king_attackers',
        'black_king_attackers',

        # Threat features (7)
        'num_attacked_pieces',
        'num_defended_pieces',
        'num_hanging_pieces',
        'control_center_squares',
        'control_key_files',
        'bishop_pair_white',
        'bishop_pair_black'
    ]


def process_dataframe(df: pd.DataFrame, fen_column: str = 'FEN') -> pd.DataFrame:
    """
    Add all 40 metadata features to a DataFrame containing FEN strings.

    Args:
        df: DataFrame with FEN column
        fen_column: Name of FEN column

    Returns:
        DataFrame with added feature columns
    """
    print(f"Extracting enhanced metadata for {len(df)} positions...")

    # Extract features for all positions
    features_list = []
    for idx, fen in enumerate(df[fen_column]):
        if idx % 10000 == 0:
            print(f"  Processed {idx}/{len(df)} positions...")

        features = extract_all_features(fen)
        features_list.append(features)

    # Create DataFrame from features
    features_df = pd.DataFrame(features_list)

    # Combine with original DataFrame
    result_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    print(f"✓ Added {len(features_df.columns)} feature columns")

    return result_df


if __name__ == '__main__':
    # Test with sample positions
    test_fens = [
        # Starting position
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        # Position with check
        "rnbqkb1r/pppp1ppp/5n2/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 3",
        # Checkmate position
        "8/5k1p/7Q/7K/5Pq1/4n3/PPP5/8 w - - 2 48",
        # Endgame position
        "8/8/8/4k3/8/8/4K3/4R3 w - - 0 1"
    ]

    print("Testing enhanced metadata extraction...\n")

    for i, fen in enumerate(test_fens, 1):
        print(f"Position {i}:")
        print(f"FEN: {fen}")

        features = extract_all_features(fen)

        print(f"  Checkmate: {features['is_checkmate']}")
        print(f"  In check: {features['is_in_check']}")
        print(f"  Legal moves: {features['num_legal_moves']:.2f}")
        print(f"  Material advantage: {features['material_advantage']:.2f}")
        print(f"  King safety (white/black): {features['white_pawn_shield']:.2f} / {features['black_pawn_shield']:.2f}")
        print()

    print(f"✓ Total features: {len(get_feature_names())}")
    print(f"Feature names: {get_feature_names()[:5]}... (showing first 5)")
