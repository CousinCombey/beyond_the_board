"""
Interactive chess board for playing and evaluating positions with Stockfish.

This script allows you to:
1. Start from an initial position (default or custom FEN)
2. Make moves interactively
3. At any state, get Stockfish evaluations for all legal moves
"""

import chess
import chess.engine
from typing import Dict, List, Optional, Tuple


class InteractiveChessEvaluator:
    """
    Interactive chess board with Stockfish evaluation capabilities.
    """

    def __init__(self, stockfish_path: str = "/usr/local/bin/stockfish",
                 initial_fen: Optional[str] = None):
        """
        Initialize the interactive chess evaluator.

        Args:
            stockfish_path: Path to Stockfish engine executable
            initial_fen: Optional FEN string for initial position (defaults to starting position)
        """
        self.board = chess.Board(initial_fen) if initial_fen else chess.Board()
        self.stockfish_path = stockfish_path
        self.move_history = []

    def get_current_fen(self) -> str:
        """Get the current board position as FEN string."""
        return self.board.fen()

    def display_board(self):
        """Display the current board state."""
        print("\n" + "="*50)
        print(self.board)
        print("="*50)
        print(f"FEN: {self.get_current_fen()}")
        print(f"Turn: {'White' if self.board.turn else 'Black'}")
        print(f"Legal moves: {self.board.legal_moves.count()}")
        print("="*50 + "\n")

    def make_move(self, move_uci: str) -> bool:
        """
        Make a move on the board.

        Args:
            move_uci: Move in UCI format (e.g., 'e2e4', 'e7e5q' for promotion)

        Returns:
            True if move was legal and made, False otherwise
        """
        try:
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.move_history.append(move_uci)
                return True
            else:
                print(f"Illegal move: {move_uci}")
                return False
        except ValueError as e:
            print(f"Invalid move format: {e}")
            return False

    def undo_move(self) -> bool:
        """
        Undo the last move.

        Returns:
            True if a move was undone, False if no moves to undo
        """
        if len(self.move_history) > 0:
            self.board.pop()
            self.move_history.pop()
            return True
        return False

    def get_legal_moves(self) -> List[str]:
        """
        Get all legal moves in the current position.

        Returns:
            List of legal moves in UCI format
        """
        return [move.uci() for move in self.board.legal_moves]

    def evaluate_position(self, time_limit: float = 0.1) -> Dict[str, any]:
        """
        Evaluate the current position with Stockfish.

        Args:
            time_limit: Time limit for analysis in seconds

        Returns:
            Dictionary with evaluation info (score, mate, etc.)
        """
        with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
            info = engine.analyse(self.board, chess.engine.Limit(time=time_limit))

            result = {
                'fen': self.get_current_fen(),
                'score': None,
                'mate': None,
                'best_move': str(info.get('pv', [None])[0]) if 'pv' in info else None
            }

            score = info.get('score')
            if score:
                # Score from white's perspective
                if score.is_mate():
                    result['mate'] = score.white().mate()
                else:
                    result['score'] = score.white().score()

            return result

    def evaluate_all_legal_moves(self, time_limit: float = 0.1) -> List[Dict[str, any]]:
        """
        Evaluate all legal moves from the current position.

        Args:
            time_limit: Time limit per move analysis in seconds

        Returns:
            List of dictionaries, each containing move and its evaluation
        """
        legal_moves = list(self.board.legal_moves)
        evaluations = []

        with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
            for move in legal_moves:
                # Make the move temporarily
                self.board.push(move)

                # Evaluate the resulting position
                info = engine.analyse(self.board, chess.engine.Limit(time=time_limit))

                eval_data = {
                    'move': move.uci(),
                    'from_square': chess.square_name(move.from_square),
                    'to_square': chess.square_name(move.to_square),
                    'resulting_fen': self.board.fen(),
                    'score': None,
                    'mate': None
                }

                score = info.get('score')
                if score:
                    # Score from white's perspective
                    if score.is_mate():
                        eval_data['mate'] = score.white().mate()
                    else:
                        eval_data['score'] = score.white().score()

                evaluations.append(eval_data)

                # Undo the move
                self.board.pop()

        # Sort by score (best for current player first)
        if self.board.turn == chess.WHITE:
            # White wants highest score
            evaluations.sort(key=lambda x: (
                x['mate'] if x['mate'] is not None else float('-inf'),
                x['score'] if x['score'] is not None else float('-inf')
            ), reverse=True)
        else:
            # Black wants lowest score
            evaluations.sort(key=lambda x: (
                -x['mate'] if x['mate'] is not None else float('inf'),
                x['score'] if x['score'] is not None else float('inf')
            ))

        return evaluations

    def print_move_evaluations(self, evaluations: List[Dict[str, any]]):
        """
        Pretty print move evaluations.

        Args:
            evaluations: List of evaluation dictionaries from evaluate_all_legal_moves
        """
        print(f"\n{'='*70}")
        print(f"EVALUATIONS FOR ALL LEGAL MOVES (from {'White' if self.board.turn else 'Black'}'s perspective)")
        print(f"{'='*70}")
        print(f"{'Rank':<6} {'Move':<8} {'From':<8} {'To':<8} {'Score':<15} {'Mate in':<10}")
        print(f"{'-'*70}")

        for idx, eval_data in enumerate(evaluations, 1):
            score_str = f"{eval_data['score']/100:+.2f}" if eval_data['score'] is not None else "N/A"
            mate_str = f"M{eval_data['mate']}" if eval_data['mate'] is not None else "-"

            print(f"{idx:<6} {eval_data['move']:<8} {eval_data['from_square']:<8} "
                  f"{eval_data['to_square']:<8} {score_str:<15} {mate_str:<10}")

        print(f"{'='*70}\n")


def interactive_session(stockfish_path: str = "/usr/local/bin/stockfish",
                       initial_fen: Optional[str] = None):
    """
    Run an interactive chess session.

    Args:
        stockfish_path: Path to Stockfish executable
        initial_fen: Optional starting position FEN
    """
    evaluator = InteractiveChessEvaluator(stockfish_path, initial_fen)

    print("\n" + "="*70)
    print("INTERACTIVE CHESS EVALUATOR")
    print("="*70)
    print("\nCommands:")
    print("  - Type a move in UCI format (e.g., 'e2e4', 'e7e8q' for promotion)")
    print("  - 'eval' - Evaluate current position")
    print("  - 'moves' - Show all legal moves")
    print("  - 'eval_all' - Evaluate all legal moves")
    print("  - 'undo' - Undo last move")
    print("  - 'fen' - Show current FEN")
    print("  - 'display' - Show current board")
    print("  - 'history' - Show move history")
    print("  - 'quit' or 'exit' - Exit the session")
    print("="*70 + "\n")

    evaluator.display_board()

    while True:
        command = input("Enter command or move: ").strip().lower()

        if command in ['quit', 'exit', 'q']:
            print("Exiting...")
            break

        elif command == 'eval':
            print("\nEvaluating position...")
            result = evaluator.evaluate_position(time_limit=0.5)
            print(f"FEN: {result['fen']}")
            if result['mate'] is not None:
                print(f"Mate in: {result['mate']}")
            elif result['score'] is not None:
                print(f"Score: {result['score']/100:+.2f} (from White's perspective)")
            print(f"Best move: {result['best_move']}\n")

        elif command == 'moves':
            moves = evaluator.get_legal_moves()
            print(f"\nLegal moves ({len(moves)}):")
            print(", ".join(moves))
            print()

        elif command == 'eval_all':
            print("\nEvaluating all legal moves (this may take a moment)...")
            evaluations = evaluator.evaluate_all_legal_moves(time_limit=0.1)
            evaluator.print_move_evaluations(evaluations)

        elif command == 'undo':
            if evaluator.undo_move():
                print("Move undone.")
                evaluator.display_board()
            else:
                print("No moves to undo.\n")

        elif command == 'fen':
            print(f"\nFEN: {evaluator.get_current_fen()}\n")

        elif command in ['display', 'd', 'show']:
            evaluator.display_board()

        elif command == 'history':
            print(f"\nMove history: {' '.join(evaluator.move_history)}\n")

        else:
            # Assume it's a move
            if evaluator.make_move(command):
                print(f"Move {command} played.")
                evaluator.display_board()
            else:
                print("Invalid command or illegal move. Type 'help' for available commands.\n")


if __name__ == "__main__":
    # Example 1: Start from initial position
    print("Starting interactive session from initial position...")
    interactive_session()

    # Example 2: Start from custom FEN
    # custom_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    # interactive_session(initial_fen=custom_fen)
