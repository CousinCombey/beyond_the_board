"""
Step 2: Self-Play Reinforcement Learning System

This system:
1. Loads Step 1 model (frozen evaluation head)
2. Adds policy head (move selection) and value head (outcome prediction)
3. Plays games against itself
4. Verifies checkmate patterns
5. Saves training data for Step 3

"""

import chess
import chess.pgn
import numpy as np
import pandas as pd
from pathlib import Path
import random
from typing import List, Tuple, Optional
from keras.models import load_model, Model
from keras import layers, Input
from beyond_the_board.tensor.main import fen_to_tensor_8_8_12
from beyond_the_board.tensor.enhanced_metadata import extract_all_features, get_feature_names


class ChessSelfPlayEngine:
    """
    Self-play engine for chess RL training.
    """

    def __init__(self, base_model_path: str, temperature: float = 1.0):
        """
        Initialize self-play engine.

        Args:
            base_model_path: Path to Step 1 trained model
            temperature: Temperature for move sampling (higher = more exploration)
        """
        self.base_model = load_model(base_model_path)
        self.temperature = temperature
        self.policy_model = self._build_policy_model()

    def _build_policy_model(self) -> Model:
        """
        Build policy network on top of Step 1 model.

        Adds:
        - Policy head: Softmax over all possible moves (4672 UCI moves)
        - Value head: Game outcome prediction

        Returns:
            Combined model with policy and value heads
        """
        # Freeze base model layers
        for layer in self.base_model.layers:
            layer.trainable = False

        # Get shared representation from base model
        # Use the layer before task-specific heads
        shared_layer = self.base_model.get_layer('shared_dense_2')
        shared_output = shared_layer.output

        # Policy head: predict move probabilities
        policy_dense = layers.Dense(1024, activation='relu', name='policy_dense_1')(shared_output)
        policy_dense = layers.Dropout(0.3, name='policy_dropout')(policy_dense)
        policy_dense = layers.Dense(512, activation='relu', name='policy_dense_2')(policy_dense)

        # Output: probability distribution over moves (simplified to 4096 for common moves)
        policy_output = layers.Dense(4096, activation='softmax', name='policy_output')(policy_dense)

        # Value head: predict game outcome
        value_dense = layers.Dense(128, activation='relu', name='value_dense')(shared_output)
        value_output = layers.Dense(1, activation='tanh', name='value_output')(value_dense)

        # Create combined model
        policy_model = Model(
            inputs=self.base_model.inputs,
            outputs=[policy_output, value_output],
            name='policy_value_model'
        )

        policy_model.compile(
            optimizer='adam',
            loss={
                'policy_output': 'categorical_crossentropy',
                'value_output': 'mean_squared_error'
            },
            loss_weights={'policy_output': 0.7, 'value_output': 0.3}
        )

        return policy_model

    def encode_position(self, board: chess.Board) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert chess board to model input format.

        Args:
            board: python-chess Board object

        Returns:
            Tuple of (board_tensor, metadata_features)
        """
        fen = board.fen()

        # Get board tensor
        board_tensor = fen_to_tensor_8_8_12(fen)
        board_tensor = np.expand_dims(board_tensor, axis=0).astype(np.float32)

        # Get metadata features
        features = extract_all_features(fen)
        feature_names = get_feature_names()
        metadata_vector = np.array([[features[name] for name in feature_names]], dtype=np.float32)

        return board_tensor, metadata_vector

    def get_legal_move_mask(self, board: chess.Board) -> np.ndarray:
        """
        Create mask for legal moves.

        Args:
            board: python-chess Board object

        Returns:
            Binary mask of shape (4096,) where 1 = legal move
        """
        mask = np.zeros(4096, dtype=np.float32)

        for i, move in enumerate(board.legal_moves):
            # Simple hash: from_square * 64 + to_square
            move_idx = move.from_square * 64 + move.to_square
            if move_idx < 4096:
                mask[move_idx] = 1.0

        return mask

    def select_move(self, board: chess.Board, use_temperature: bool = True) -> chess.Move:
        """
        Select move using policy network.

        Args:
            board: Current board position
            use_temperature: Whether to use temperature for exploration

        Returns:
            Selected chess move
        """
        if board.is_game_over():
            return None

        # Get model inputs
        board_tensor, metadata = self.encode_position(board)

        # Get policy prediction
        policy_probs, value = self.policy_model.predict([board_tensor, metadata], verbose=0)
        policy_probs = policy_probs[0]

        # Apply legal move mask
        legal_mask = self.get_legal_move_mask(board)
        legal_probs = policy_probs * legal_mask

        # Normalize
        if legal_probs.sum() > 0:
            legal_probs = legal_probs / legal_probs.sum()
        else:
            # Fallback: uniform over legal moves
            legal_probs = legal_mask / legal_mask.sum()

        # Apply temperature
        if use_temperature and self.temperature != 1.0:
            legal_probs = np.power(legal_probs, 1.0 / self.temperature)
            legal_probs = legal_probs / legal_probs.sum()

        # Sample move
        move_idx = np.random.choice(len(legal_probs), p=legal_probs)

        # Convert index back to move
        from_square = move_idx // 64
        to_square = move_idx % 64

        # Find matching legal move
        for move in board.legal_moves:
            if move.from_square == from_square and move.to_square == to_square:
                return move

        # Fallback: random legal move
        return random.choice(list(board.legal_moves))

    def play_game(self, start_fen: Optional[str] = None, max_moves: int = 200) -> dict:
        """
        Play a complete game and record trajectory.

        Args:
            start_fen: Starting position (None = standard start)
            max_moves: Maximum moves before draw

        Returns:
            Dictionary with game data
        """
        board = chess.Board(start_fen) if start_fen else chess.Board()

        trajectory = []
        move_count = 0

        while not board.is_game_over() and move_count < max_moves:
            # Record position before move
            fen_before = board.fen()

            # Get evaluation from base model
            board_tensor, metadata = self.encode_position(board)
            eval_pred = self.base_model.predict([board_tensor, metadata], verbose=0)
            eval_score = float(eval_pred[0][0][0])  # evaluation_output

            # Select and make move
            move = self.select_move(board)
            if move is None:
                break

            move_san = board.san(move)
            board.push(move)

            # Record position after move
            fen_after = board.fen()

            # Check if this move led to checkmate
            is_checkmate = board.is_checkmate()
            is_check = board.is_check()

            trajectory.append({
                'fen_before': fen_before,
                'move_san': move_san,
                'move_uci': move.uci(),
                'fen_after': fen_after,
                'eval_before': eval_score,
                'move_number': move_count + 1,
                'is_check': is_check,
                'is_checkmate': is_checkmate
            })

            move_count += 1

        # Determine game result
        if board.is_checkmate():
            result = '1-0' if board.turn == chess.BLACK else '0-1'
            termination = 'checkmate'
        elif board.is_stalemate():
            result = '1/2-1/2'
            termination = 'stalemate'
        elif board.is_insufficient_material():
            result = '1/2-1/2'
            termination = 'insufficient_material'
        elif move_count >= max_moves:
            result = '1/2-1/2'
            termination = 'max_moves'
        else:
            result = '1/2-1/2'
            termination = 'draw'

        # Add result to all positions
        for pos in trajectory:
            pos['game_result'] = result
            pos['termination'] = termination

        return {
            'trajectory': trajectory,
            'result': result,
            'termination': termination,
            'num_moves': move_count,
            'final_fen': board.fen()
        }

    def verify_checkmate_patterns(self, game_data: dict) -> dict:
        """
        Verify checkmate patterns and identify missed mates.

        Args:
            game_data: Game data from play_game()

        Returns:
            Updated game data with checkmate analysis
        """
        trajectory = game_data['trajectory']

        for i, position in enumerate(trajectory):
            board = chess.Board(position['fen_before'])

            # Check if there's a mate in 1
            mate_in_1 = None
            for move in board.legal_moves:
                board_copy = board.copy()
                board_copy.push(move)
                if board_copy.is_checkmate():
                    mate_in_1 = move.uci()
                    break

            position['mate_in_1_available'] = mate_in_1 is not None
            position['mate_in_1_move'] = mate_in_1
            position['found_mate'] = position['is_checkmate'] and mate_in_1 is not None
            position['missed_mate'] = mate_in_1 is not None and not position['is_checkmate']

        game_data['trajectory'] = trajectory
        game_data['total_mates_found'] = sum(1 for p in trajectory if p.get('found_mate', False))
        game_data['total_mates_missed'] = sum(1 for p in trajectory if p.get('missed_mate', False))

        return game_data


def run_self_play_session(base_model_path: str, num_games: int = 100,
                          output_csv: str = 'self_play_games.csv',
                          starting_positions: Optional[List[str]] = None):
    """
    Run complete self-play session.

    Args:
        base_model_path: Path to Step 1 model
        num_games: Number of games to play
        output_csv: Path to save game data
        starting_positions: List of FENs to start from (None = random)

    Returns:
        DataFrame with all game positions
    """
    print("=" * 80)
    print("STEP 2: SELF-PLAY RL TRAINING")
    print("=" * 80)

    engine = ChessSelfPlayEngine(base_model_path, temperature=1.2)

    all_positions = []

    for game_num in range(num_games):
        if game_num % 10 == 0:
            print(f"\nPlaying game {game_num+1}/{num_games}...")

        # Select starting position
        start_fen = None
        if starting_positions:
            start_fen = random.choice(starting_positions)

        # Play game
        game_data = engine.play_game(start_fen=start_fen)

        # Verify checkmates
        game_data = engine.verify_checkmate_patterns(game_data)

        # Add game ID to all positions
        for position in game_data['trajectory']:
            position['game_id'] = game_num
            all_positions.append(position)

        if game_num % 10 == 9:
            print(f"  Game {game_num+1}: {game_data['result']} after {game_data['num_moves']} moves")
            print(f"  Termination: {game_data['termination']}")
            print(f"  Mates found: {game_data['total_mates_found']}, Missed: {game_data['total_mates_missed']}")

    # Convert to DataFrame
    df = pd.DataFrame(all_positions)

    # Save to CSV
    df.to_csv(output_csv, index=False)

    print("\n" + "=" * 80)
    print("SELF-PLAY SESSION COMPLETE")
    print("=" * 80)
    print(f"Total games: {num_games}")
    print(f"Total positions: {len(df)}")
    print(f"Checkmate games: {df['is_checkmate'].sum()}")
    print(f"Check positions: {df['is_check'].sum()}")
    print(f"Missed mates: {df['missed_mate'].sum()}")
    print(f"Data saved to: {output_csv}")

    return df


if __name__ == '__main__':
    print("Step 2 Self-Play RL System")
    print("\nThis module requires Step 1 trained model.")
    print("Usage:")
    print("  from beyond_the_board.models.step2_selfplay_rl import run_self_play_session")
    print("  df = run_self_play_session('models/step1_model.keras', num_games=100)")
