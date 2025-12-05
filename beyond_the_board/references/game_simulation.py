import random
import chess
import chess.svg
import chess.pgn
import chess.engine
from beyond_the_board.models.cnn_john import model_predict
from IPython.display import display, SVG, clear_output
import ipywidgets as widgets

def show_moves_interactive(moves_list, result=None, termination=None):
    """
    Display chess moves interactively in Jupyter notebook.

    Features:
    - Navigate through moves using a slider
    - Arrows show each move (white for white's moves, black for black's)
    - Displays move descriptions in chess notation
    - Shows game-ending messages (checkmate, stalemate, etc.)

    Args:
        moves_list: List of moves in UCI format (e.g., ['e2e4', 'e7e5'])
        result: Game result ('1-0', '0-1', '1/2-1/2') - optional
        termination: How the game ended - optional
    """

    def show_position(move_idx):
        """Display board position at given move index."""
        board = chess.Board()

        # Replay all moves up to this point
        move_description = ""
        for i in range(move_idx):
            if i < len(moves_list):
                # Build description for the last move
                if i == move_idx - 1:
                    move = chess.Move.from_uci(moves_list[i])
                    piece = board.piece_at(move.from_square)

                    # Get piece name and color
                    piece_name = chess.piece_name(piece.piece_type).title()
                    color = "White" if piece.color == chess.WHITE else "Black"

                    # Check move type
                    is_capture = board.is_capture(move)
                    is_castling = board.is_castling(move)

                    # Get square names
                    from_square = chess.square_name(move.from_square)
                    to_square = chess.square_name(move.to_square)

                    # Build description
                    move_number = (i // 2) + 1
                    if is_castling:
                        if to_square in ['g1', 'g8']:
                            move_description = f"Move {move_number}: {color} castles kingside"
                        else:
                            move_description = f"Move {move_number}: {color} castles queenside"
                    elif is_capture:
                        captured_piece = board.piece_at(move.to_square)
                        captured_name = chess.piece_name(captured_piece.piece_type).title()
                        move_description = f"Move {move_number}: {color} {piece_name} takes {captured_name} on {to_square}"
                    else:
                        move_description = f"Move {move_number}: {color} {piece_name} to {to_square}"

                board.push(chess.Move.from_uci(moves_list[i]))

        # Generate board visualization with arrow
        if move_idx > 0 and move_idx <= len(moves_list):
            move = chess.Move.from_uci(moves_list[move_idx - 1])

            # Arrow color based on whose move it was
            # Even indices (0, 2, 4...) = white's moves
            # Odd indices (1, 3, 5...) = black's moves
            arrow_color = 'white' if (move_idx - 1) % 2 == 0 else 'black'

            arrow = chess.svg.Arrow(move.from_square, move.to_square, color=arrow_color)
            svg = chess.svg.board(board, size=400, arrows=[arrow])
        else:
            svg = chess.svg.board(board, size=400)
            move_description = "Starting position"

        clear_output(wait=True)
        print(f"Position {move_idx} of {len(moves_list)}")
        print(f"\n{move_description}\n")

        # Check for game-ending conditions at the last move
        if move_idx == len(moves_list) and move_idx > 0:
            game_end_message = ""

            # Automatic detection from board state
            if board.is_checkmate():
                winner = "Black" if board.turn == chess.WHITE else "White"
                game_end_message = f"🏁 CHECKMATE! {winner} wins!"
            elif board.is_stalemate():
                game_end_message = "🏁 STALEMATE! Game is a draw."
            elif board.is_insufficient_material():
                game_end_message = "🏁 DRAW by insufficient material."
            elif board.is_fifty_moves():
                game_end_message = "🏁 DRAW by fifty-move rule."
            elif board.is_repetition():
                game_end_message = "🏁 DRAW by threefold repetition."
            elif board.is_variant_draw():
                game_end_message = "🏁 DRAW by variant rules."

            # Use termination reason from dataset if available
            if termination:
                if not game_end_message:  # No automatic detection
                    if termination.lower() in ['time forfeit', 'abandoned']:
                        game_end_message = f"🏁 GAME ENDED: {termination}"
                        if result:
                            if result == '1-0':
                                game_end_message += " - White wins"
                            elif result == '0-1':
                                game_end_message += " - Black wins"
                            elif result == '1/2-1/2':
                                game_end_message += " - Draw"
                    else:
                        game_end_message = f"🏁 GAME ENDED: {termination}"

            if game_end_message:
                print(f"\n{game_end_message}\n")

        display(SVG(svg))

    # Create interactive slider widget
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(moves_list),
        step=1,
        description='Move:',
        continuous_update=False
    )

    widgets.interact(show_position, move_idx=slider)


def simulate_game(model, initial_fen):

    model = load_model()

    board = chess.Board(fen=initial_fen)
    len_legal_moves = len(list(board.legal_moves))


    #evaluate base stockfish on the initial board
    best_stockfish = model_predict(model, initial_fen)
    fen = initial_fen
    best_fen = fen
    first_move = random.choice(list(board.legal_moves))
    board.push(first_move)

    best_fen = board.fen()

    moves_list = [str(first_move)]

    j = 0
    #loop through each move and evaluate the stockfish of the fen generated
    while j < 150  or len_legal_moves ==0 :
        #initialize board at specific turn
        fen = best_fen
        board = chess.Board(fen=fen)
        len_legal_moves = len(list(board.legal_moves))

        # initiate empty lists
        possible_moves = []
        possible_fens = []

        for i in range(len_legal_moves) :
            # reinitialize the board on each loop otherwise the legal moves change
            board = chess.Board(fen=fen)
            legal_moves = list(board.legal_moves)

            #choose move and apply it to the board to get a new fen
            move = legal_moves[i]
            board.push(move)
            new_fen = board.fen()
            # #evaluate the stockfish
            # new_stockfish = model_predict(model, new_fen)
            #add fens & stockfish to lists
            possible_moves.append(str(move))
            possible_fens.append(new_fen)

        stockfish_possible_scores = batch_predict(model, possible_fens)

        if " w " in fen :

            arg_max = stockfish_possible_scores.argmax()
            best_fen = possible_fens[arg_max]
            best_stockfish = stockfish_possible_scores[arg_max]
            chosen_move = possible_moves[arg_max]
            moves_list.append(chosen_move)
            who_moved = " w "

        elif " b " in fen :
            arg_min = stockfish_possible_scores.argmin()
            best_fen = possible_fens[arg_min]
            best_stockfish = stockfish_possible_scores[arg_min]
            chosen_move = possible_moves[arg_min]
            moves_list.append(chosen_move)
            to_move = " b "
        j+=1

    return show_moves_interactive(moves_list)
