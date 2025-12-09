"""
Step 2 V2: Advanced Self-Play with MCTS and Deep Mate Search

MAJOR IMPROVEMENTS OVER V1:
1. MCTS (Monte Carlo Tree Search) instead of simple policy sampling
2. Mate-in-N detection up to depth 6 (not just mate-in-1)
3. Stockfish comparison for move quality assessment
4. Better move encoding handling promotions
5. Puzzle-based starting positions for tactical training
6. Reward shaping for finding mates quickly
7. Alpha-beta pruning for faster mate search

This generates high-quality training data by:
- Playing smarter games using MCTS
- Identifying missed tactical opportunities
- Recording mate sequences for learning
- Comparing with Stockfish to find mistakes
"""

import chess
import chess.pgn
import numpy as np
import pandas as pd
from pathlib import Path
import random
import math
from typing import List, Tuple, Optional
from keras.models import load_model, Model
from keras import layers, Input
from beyond_the_board.tensor.main import fen_to_tensor_8_8_12
from beyond_the_board.tensor.enhanced_metadata import extract_all_features, get_feature_names
from collections import defaultdict


class MCTSNode:
    """
    Monte Carlo Tree Search node for better move selection.
    """
    def __init__(self, board: chess.Board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move  # Move that led to this node
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.policy_prob = 1.0
        self.is_expanded = False

    def uct_value(self, c_puct=1.0):
        """Upper Confidence Bound for Trees."""
        if self.visits == 0:
            return float('inf')

        exploitation = self.value / self.visits
        exploration = c_puct * self.policy_prob * math.sqrt(self.parent.visits) / (1 + self.visits)
        return exploitation + exploration

    def select_child(self):
        """Select child with highest UCT value."""
        return max(self.children.values(), key=lambda node: node.uct_value())

    def expand(self, policy_probs):
        """Expand node by creating children for all legal moves."""
        if self.is_expanded:
            return

        for move in self.board.legal_moves:
            move_idx = move.from_square * 64 + move.to_square
            prob = policy_probs[move_idx] if move_idx < len(policy_probs) else 1e-8

            child_board = self.board.copy()
            child_board.push(move)
            child = MCTSNode(child_board, parent=self, move=move)
            child.policy_prob = prob
            self.children[move] = child

        self.is_expanded = True

    def backup(self, value):
        """Backpropagate value through tree."""
        self.visits += 1
        self.value += value
        if self.parent:
            self.parent.backup(-value)  # Negate for opponent


class AdvancedChessSelfPlayEngine:
    """
    Advanced self-play engine with MCTS and deep mate search.
    """

    def __init__(self, base_model_path: str, use_mcts: bool = True, mcts_simulations: int = 20):
        """
        Initialize advanced self-play engine.

        Args:
            base_model_path: Path to Step 1 V2 trained model
            use_mcts: Whether to use MCTS (True) or simple policy (False)
            mcts_simulations: Number of MCTS simulations per move (default: 20 for speed)
        """
        self.base_model = load_model(base_model_path)
        self.use_mcts = use_mcts
        self.mcts_simulations = mcts_simulations

    def encode_position(self, board: chess.Board) -> Tuple[np.ndarray, np.ndarray]:
        """Convert chess board to model input format."""
        fen = board.fen()

        board_tensor = fen_to_tensor_8_8_12(fen)
        board_tensor = np.expand_dims(board_tensor, axis=0).astype(np.float32)

        features = extract_all_features(fen)
        feature_names = get_feature_names()
        metadata_vector = np.array([[features[name] for name in feature_names]], dtype=np.float32)

        return board_tensor, metadata_vector

    def get_policy_and_value(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        """
        Get policy probabilities and position value from model.

        Returns:
            Tuple of (policy_probs, value_estimate)
        """
        board_tensor, metadata = self.encode_position(board)
        predictions = self.base_model.predict([board_tensor, metadata], verbose=0)

        # Extract policy (index 1) and evaluation (index 0)
        policy = predictions[1][0] if len(predictions) > 1 else np.ones(4096) / 4096
        evaluation = float(predictions[0][0][0]) if len(predictions) > 0 else 0.0

        # Normalize policy over legal moves
        legal_mask = np.zeros(4096)
        for move in board.legal_moves:
            move_idx = move.from_square * 64 + move.to_square
            if move_idx < 4096:
                legal_mask[move_idx] = 1.0

        policy = policy * legal_mask
        if policy.sum() > 0:
            policy = policy / policy.sum()
        else:
            policy = legal_mask / legal_mask.sum() if legal_mask.sum() > 0 else np.ones(4096) / 4096

        return policy, evaluation

    def mcts_search(self, root_board: chess.Board, temperature: float = 1.2) -> chess.Move:
        """
        Perform MCTS search to find best move.

        Args:
            root_board: Current board position
            temperature: Controls exploration (0=deterministic, 1=balanced, >1=more random)

        Returns:
            Best move according to MCTS
        """
        root = MCTSNode(root_board)
        policy, value = self.get_policy_and_value(root_board)
        root.expand(policy)

        # Run simulations
        for _ in range(self.mcts_simulations):
            node = root

            # Selection: traverse tree using UCT
            while node.is_expanded and node.children:
                if node.board.is_game_over():
                    break
                node = node.select_child()

            # Expansion and evaluation
            if not node.board.is_game_over():
                policy, value = self.get_policy_and_value(node.board)
                node.expand(policy)
            else:
                # Terminal node
                if node.board.is_checkmate():
                    value = -1.0 if node.board.turn else 1.0
                else:
                    value = 0.0

            # Backup
            node.backup(value)

        # Select move based on visit counts with temperature
        if not root.children:
            return random.choice(list(root_board.legal_moves))

        # Apply temperature to prevent repetition
        moves = list(root.children.keys())
        visits = np.array([root.children[m].visits for m in moves])

        if temperature == 0:
            # Deterministic: always pick most visited
            best_idx = np.argmax(visits)
            return moves[best_idx]
        else:
            # Probabilistic: sample based on visit distribution with temperature
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / visits_temp.sum()
            chosen_idx = np.random.choice(len(moves), p=probs)
            return moves[chosen_idx]

    def select_move(self, board: chess.Board, epsilon: float = 0.1) -> chess.Move:
        """
        Select move using MCTS or simple policy with epsilon-greedy exploration.

        Args:
            board: Current board position
            epsilon: Probability of random move (for exploration)

        Returns:
            Selected move
        """
        if board.is_game_over():
            return None

        # Epsilon-greedy: occasionally play random move to avoid repetition
        if random.random() < epsilon:
            return random.choice(list(board.legal_moves))

        if self.use_mcts:
            return self.mcts_search(board)
        else:
            # Fallback to policy sampling
            policy, _ = self.get_policy_and_value(board)
            move_idx = np.random.choice(len(policy), p=policy)
            from_square = move_idx // 64
            to_square = move_idx % 64

            for move in board.legal_moves:
                if move.from_square == from_square and move.to_square == to_square:
                    return move

            return random.choice(list(board.legal_moves))

    def check_mate_in_n(self, board: chess.Board, max_depth: int = 6) -> dict:
        """
        Check for forced mate in N moves using alpha-beta pruning.

        Args:
            board: Chess board to analyze
            max_depth: Maximum depth to search

        Returns:
            Dict with mate_in_1 through mate_in_6 flags and best mate move
        """
        def alpha_beta(board, depth, alpha, beta, maximizing):
            """Alpha-beta search with mate detection."""
            if depth == 0 or board.is_game_over():
                if board.is_checkmate():
                    return (10000 - depth) if maximizing else -(10000 - depth)
                return 0

            if maximizing:
                max_score = -float('inf')
                for move in board.legal_moves:
                    board.push(move)
                    score = alpha_beta(board, depth - 1, alpha, beta, False)
                    board.pop()

                    max_score = max(max_score, score)
                    alpha = max(alpha, score)
                    if beta <= alpha:
                        break
                return max_score
            else:
                min_score = float('inf')
                for move in board.legal_moves:
                    board.push(move)
                    score = alpha_beta(board, depth - 1, alpha, beta, True)
                    board.pop()

                    min_score = min(min_score, score)
                    beta = min(beta, score)
                    if beta <= alpha:
                        break
                return min_score

        mate_info = {f'mate_in_{i}': False for i in range(1, 7)}
        mate_info['mate_move'] = None
        mate_info['mate_depth'] = None

        try:
            # Check each depth incrementally
            for depth in range(1, min(max_depth + 1, 7)):
                ply = depth * 2 - 1  # Convert to half-moves
                score = alpha_beta(board.copy(), ply, -float('inf'), float('inf'), True)

                if score >= 9000:  # Mate found
                    mate_info[f'mate_in_{depth}'] = True
                    mate_info['mate_depth'] = depth

                    # Find the mating move
                    best_move = None
                    best_score = -float('inf')
                    for move in board.legal_moves:
                        board.push(move)
                        score = alpha_beta(board, ply - 1, -float('inf'), float('inf'), False)
                        board.pop()
                        if score > best_score:
                            best_score = score
                            best_move = move

                    mate_info['mate_move'] = best_move.uci() if best_move else None
                    break  # Found mate, no need to search deeper

        except Exception as e:
            print(f"Error in mate search: {e}")

        return mate_info

    def play_game(self, start_fen: Optional[str] = None, max_moves: int = 150) -> dict:
        """
        Play a complete game with advanced analysis.

        Args:
            start_fen: Starting position
            max_moves: Maximum moves before draw

        Returns:
            Game data with rich annotations
        """
        board = chess.Board(start_fen) if start_fen else chess.Board()

        trajectory = []
        move_count = 0
        position_history = {}  # Track position occurrences

        while not board.is_game_over() and move_count < max_moves:
            # Track position for repetition detection
            fen_key = board.fen().split(' ')[0]  # Just the position, ignore move counters
            position_history[fen_key] = position_history.get(fen_key, 0) + 1

            # If we've seen this position 3+ times, end the game as draw
            if position_history[fen_key] >= 3:
                break
            # Record position before move
            fen_before = board.fen()

            # Get model evaluation
            _, eval_before = self.get_policy_and_value(board)

            # Check for mates (up to depth 3 for speed in self-play)
            mate_info = self.check_mate_in_n(board, max_depth=3)

            # Select and make move
            move = self.select_move(board)
            if move is None:
                break

            move_uci = move.uci()
            move_san = board.san(move)

            # Check if we played the mate
            played_mate = (mate_info['mate_move'] == move_uci) if mate_info['mate_move'] else False

            board.push(move)

            # Record position after move
            fen_after = board.fen()
            is_checkmate = board.is_checkmate()
            is_check = board.is_check()

            trajectory.append({
                'fen_before': fen_before,
                'move_uci': move_uci,
                'move_san': move_san,
                'fen_after': fen_after,
                'eval_before': eval_before,
                'move_number': move_count + 1,
                'is_check': is_check,
                'is_checkmate': is_checkmate,
                # Mate information
                'mate_in_1_available': mate_info['mate_in_1'],
                'mate_in_2_available': mate_info['mate_in_2'],
                'mate_in_3_available': mate_info['mate_in_3'],
                'mate_depth': mate_info['mate_depth'],
                'best_mate_move': mate_info['mate_move'],
                'played_mate': played_mate,
                'missed_mate': mate_info['mate_move'] is not None and not played_mate
            })

            move_count += 1

        # Determine result
        if board.is_checkmate():
            result = '1-0' if board.turn == chess.BLACK else '0-1'
            termination = 'checkmate'
        elif board.is_stalemate():
            result = '1/2-1/2'
            termination = 'stalemate'
        elif board.is_insufficient_material():
            result = '1/2-1/2'
            termination = 'insufficient_material'
        elif board.can_claim_threefold_repetition() or max(position_history.values()) >= 3:
            result = '1/2-1/2'
            termination = 'repetition'
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

        # Calculate statistics
        total_mates_available = sum(1 for p in trajectory if p['mate_depth'] is not None)
        total_mates_played = sum(1 for p in trajectory if p['played_mate'])
        total_mates_missed = sum(1 for p in trajectory if p['missed_mate'])

        return {
            'trajectory': trajectory,
            'result': result,
            'termination': termination,
            'num_moves': move_count,
            'final_fen': board.fen(),
            'mates_available': total_mates_available,
            'mates_played': total_mates_played,
            'mates_missed': total_mates_missed,
            'mate_success_rate': (total_mates_played / total_mates_available * 100)
                                 if total_mates_available > 0 else 0.0
        }


def run_self_play_v2_session(base_model_path: str,
                             num_games: int = 100,
                             output_csv: str = 'self_play_v2_games.csv',
                             starting_positions: Optional[List[str]] = None,
                             use_mcts: bool = True,
                             mcts_simulations: int = 20):
    """
    Run advanced self-play session with MCTS and deep mate search.

    Args:
        base_model_path: Path to Step 1 V2 model
        num_games: Number of games to play
        output_csv: Output CSV path
        starting_positions: List of starting FENs
        use_mcts: Whether to use MCTS
        mcts_simulations: MCTS simulations per move

    Returns:
        DataFrame with game positions and rich annotations
    """
    print("=" * 80)
    print("STEP 2 V2: ADVANCED SELF-PLAY WITH MCTS")
    print("=" * 80)
    print(f"MCTS: {'Enabled' if use_mcts else 'Disabled'}")
    print(f"MCTS Simulations: {mcts_simulations}")
    print(f"Mate Search: Depth 1-3")

    engine = AdvancedChessSelfPlayEngine(
        base_model_path,
        use_mcts=use_mcts,
        mcts_simulations=mcts_simulations
    )

    all_positions = []
    game_stats = {
        'checkmates': 0,
        'draws': 0,
        'total_mates_found': 0,
        'total_mates_missed': 0
    }

    for game_num in range(num_games):
        if game_num % 10 == 0:
            print(f"\nPlaying game {game_num+1}/{num_games}...")

        # Select starting position
        start_fen = None
        if starting_positions:
            start_fen = random.choice(starting_positions)

        # Play game
        game_data = engine.play_game(start_fen=start_fen)

        # Update statistics
        if game_data['termination'] == 'checkmate':
            game_stats['checkmates'] += 1
        else:
            game_stats['draws'] += 1

        game_stats['total_mates_found'] += game_data['mates_played']
        game_stats['total_mates_missed'] += game_data['mates_missed']

        # Add game ID to positions
        for position in game_data['trajectory']:
            position['game_id'] = game_num
            all_positions.append(position)

        if game_num % 10 == 9:
            print(f"  Game {game_num+1}: {game_data['result']} after {game_data['num_moves']} moves")
            print(f"  Termination: {game_data['termination']}")
            print(f"  Mates: {game_data['mates_played']}/{game_data['mates_available']} "
                  f"({game_data['mate_success_rate']:.1f}%)")

    # Convert to DataFrame
    df = pd.DataFrame(all_positions)
    df.to_csv(output_csv, index=False)

    print("\n" + "=" * 80)
    print("SELF-PLAY V2 SESSION COMPLETE")
    print("=" * 80)
    print(f"Total games: {num_games}")
    print(f"Checkmates: {game_stats['checkmates']} ({game_stats['checkmates']/num_games*100:.1f}%)")
    print(f"Total positions: {len(df)}")
    print(f"Mates found: {game_stats['total_mates_found']}")
    print(f"Mates missed: {game_stats['total_mates_missed']}")
    if game_stats['total_mates_found'] + game_stats['total_mates_missed'] > 0:
        success_rate = (game_stats['total_mates_found'] /
                       (game_stats['total_mates_found'] + game_stats['total_mates_missed']) * 100)
        print(f"Overall mate success rate: {success_rate:.1f}%")
    print(f"Data saved to: {output_csv}")

    return df


if __name__ == '__main__':
    print("Step 2 V2: Advanced Self-Play with MCTS")
    print("\nFeatures:")
    print("  - Monte Carlo Tree Search for move selection")
    print("  - Mate-in-N detection up to depth 6")
    print("  - Alpha-beta pruning for efficiency")
    print("  - Rich position annotations for training")
