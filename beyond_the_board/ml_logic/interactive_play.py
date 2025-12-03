"""
Fully interactive chess board for Jupyter notebooks with instant Stockfish evaluation.

Features:
- Visual chess board
- Clickable move selection
- Instant evaluation display
- Undo/Reset functionality
"""

import chess
import chess.engine
import chess.svg
from IPython.display import display, SVG, clear_output, HTML
import ipywidgets as widgets
from typing import Optional, List, Dict


class InteractiveChessBoard:
    """
    Fully interactive chess board with Stockfish evaluation in Jupyter.
    """

    def __init__(self, stockfish_path: str = "/usr/local/bin/stockfish",
                 initial_fen: Optional[str] = None,
                 eval_time: float = 0.1):
        """
        Initialize interactive chess board.

        Args:
            stockfish_path: Path to Stockfish executable
            initial_fen: Optional starting FEN position
            eval_time: Time limit per move evaluation in seconds
        """
        self.stockfish_path = stockfish_path
        self.eval_time = eval_time
        self.initial_fen = initial_fen or chess.Board().fen()
        self.board = chess.Board(self.initial_fen)
        self.move_history = []
        self.evaluations = []

        # Create widgets
        self.board_output = widgets.Output()
        self.moves_output = widgets.Output()
        self.info_output = widgets.Output()

        # Create buttons
        self.undo_button = widgets.Button(
            description='⬅️ Undo Move',
            button_style='warning',
            layout=widgets.Layout(width='150px')
        )
        self.reset_button = widgets.Button(
            description='🔄 Reset Board',
            button_style='danger',
            layout=widgets.Layout(width='150px')
        )
        self.eval_button = widgets.Button(
            description='🔍 Evaluate All',
            button_style='info',
            layout=widgets.Layout(width='150px')
        )

        # Button callbacks
        self.undo_button.on_click(self._on_undo)
        self.reset_button.on_click(self._on_reset)
        self.eval_button.on_click(self._on_evaluate)

        # Auto-evaluate on start
        self._evaluate_all_moves()
        self._update_display()

    def _evaluate_all_moves(self):
        """Evaluate all legal moves from current position."""
        legal_moves = list(self.board.legal_moves)
        self.evaluations = []

        if not legal_moves:
            return

        with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
            for move in legal_moves:
                # Get SAN and piece info before making the move
                san_notation = self.board.san(move)
                piece = self.board.piece_at(move.from_square)
                piece_symbol = piece.unicode_symbol() if piece else ""

                # Make move temporarily
                self.board.push(move)

                # Evaluate resulting position
                info = engine.analyse(self.board, chess.engine.Limit(time=self.eval_time))

                eval_data = {
                    'move': move.uci(),
                    'san': san_notation,
                    'piece_symbol': piece_symbol,
                    'resulting_fen': self.board.fen(),
                    'score': None,
                    'mate': None,
                    'display_score': 'N/A'
                }

                score = info.get('score')
                if score:
                    # Score from white's perspective
                    if score.is_mate():
                        mate_in = score.white().mate()
                        eval_data['mate'] = mate_in
                        eval_data['display_score'] = f"M{mate_in}"
                    else:
                        centipawns = score.white().score()
                        eval_data['score'] = centipawns
                        eval_data['display_score'] = f"{centipawns/100:+.2f}"

                self.evaluations.append(eval_data)

                # Undo move
                self.board.pop()

        # Sort by best move for current player
        if self.board.turn == chess.WHITE:
            # White wants highest score
            self.evaluations.sort(
                key=lambda x: (
                    x['mate'] if x['mate'] is not None else float('-inf'),
                    x['score'] if x['score'] is not None else float('-inf')
                ),
                reverse=True
            )
        else:
            # Black wants lowest score
            self.evaluations.sort(
                key=lambda x: (
                    -x['mate'] if x['mate'] is not None else float('inf'),
                    x['score'] if x['score'] is not None else float('inf')
                )
            )

    def _make_move_callback(self, move_uci: str):
        """Callback for when a move button is clicked."""
        def callback(button):
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.move_history.append(move_uci)
                self._evaluate_all_moves()
                self._update_display()
        return callback

    def _update_display(self):
        """Update all display components."""
        # Update board
        with self.board_output:
            clear_output(wait=True)
            # Add arrow for last move
            arrows = []
            if self.move_history:
                last_move = chess.Move.from_uci(self.move_history[-1])
                arrow_color = 'blue' if len(self.move_history) % 2 == 1 else 'red'
                arrows = [chess.svg.Arrow(last_move.from_square, last_move.to_square, color=arrow_color)]

            svg = chess.svg.board(self.board, size=500, arrows=arrows)
            display(SVG(svg))

        # Update info
        with self.info_output:
            clear_output(wait=True)
            turn = "White" if self.board.turn else "Black"
            print(f"Turn: {turn}")
            print(f"FEN: {self.board.fen()}")
            print(f"Moves played: {len(self.move_history)}")

            if self.board.is_checkmate():
                winner = "Black" if self.board.turn == chess.WHITE else "White"
                print(f"\n🏆 CHECKMATE! {winner} wins!")
            elif self.board.is_stalemate():
                print("\n🤝 STALEMATE - Draw!")
            elif self.board.is_check():
                print(f"\n⚠️ {turn} is in CHECK!")

        # Update moves list
        with self.moves_output:
            clear_output(wait=True)

            if not self.evaluations:
                print("No legal moves available.")
                return

            # Display moves as buttons with evaluations
            print(f"\n{'='*60}")
            print(f"Legal Moves for {turn} (Best to Worst):")
            print(f"{'='*60}\n")

            buttons = []
            for idx, eval_data in enumerate(self.evaluations, 1):
                # Color code based on evaluation
                if eval_data['mate'] is not None:
                    if (self.board.turn == chess.WHITE and eval_data['mate'] > 0) or \
                       (self.board.turn == chess.BLACK and eval_data['mate'] < 0):
                        style = 'success'  # Good mate
                    else:
                        style = 'danger'  # Bad mate
                elif eval_data['score'] is not None:
                    score = eval_data['score']
                    if self.board.turn == chess.WHITE:
                        if score > 100:
                            style = 'success'
                        elif score > 0:
                            style = 'info'
                        elif score > -100:
                            style = 'warning'
                        else:
                            style = 'danger'
                    else:  # Black's turn
                        if score < -100:
                            style = 'success'
                        elif score < 0:
                            style = 'info'
                        elif score < 100:
                            style = 'warning'
                        else:
                            style = 'danger'
                else:
                    style = ''

                # Use piece symbol stored during evaluation
                piece_symbol = eval_data['piece_symbol']

                button_label = f"{idx}. {piece_symbol} {eval_data['move']} [{eval_data['display_score']}]"

                button = widgets.Button(
                    description=button_label,
                    button_style=style,
                    layout=widgets.Layout(width='300px', margin='2px')
                )
                button.on_click(self._make_move_callback(eval_data['move']))
                buttons.append(button)

            # Display buttons in grid
            grid = widgets.GridBox(
                buttons,
                layout=widgets.Layout(
                    grid_template_columns='repeat(2, 300px)',
                    grid_gap='5px'
                )
            )
            display(grid)

    def _on_undo(self, button):
        """Handle undo button click."""
        if len(self.move_history) > 0:
            self.board.pop()
            self.move_history.pop()
            self._evaluate_all_moves()
            self._update_display()

    def _on_reset(self, button):
        """Handle reset button click."""
        self.board = chess.Board(self.initial_fen)
        self.move_history = []
        self._evaluate_all_moves()
        self._update_display()

    def _on_evaluate(self, button):
        """Handle evaluate button click."""
        self._evaluate_all_moves()
        self._update_display()

    def display(self):
        """Display the interactive chess board."""
        # Control buttons
        controls = widgets.HBox([
            self.undo_button,
            self.reset_button,
            self.eval_button
        ])

        # Main layout
        main_layout = widgets.VBox([
            widgets.HTML("<h2>♟️ Interactive Chess Board with Stockfish Evaluation</h2>"),
            controls,
            widgets.HBox([
                self.board_output,
                widgets.VBox([self.info_output, self.moves_output])
            ])
        ])

        display(main_layout)


def create_interactive_board(stockfish_path: str = "/usr/local/bin/stockfish",
                             initial_fen: Optional[str] = None,
                             eval_time: float = 0.1):
    """
    Create and display an interactive chess board.

    Args:
        stockfish_path: Path to Stockfish executable
        initial_fen: Optional starting FEN position
        eval_time: Time per move evaluation in seconds

    Returns:
        InteractiveChessBoard instance
    """
    board = InteractiveChessBoard(stockfish_path, initial_fen, eval_time)
    board.display()
    return board
