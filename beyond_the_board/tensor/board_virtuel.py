import chess

FEN_new_game = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

def new_game() -> object:
    """ initialiser un plateau de début de partie """
    return chess.Board()

def fen_to_board(FEN = FEN_new_game: str) -> object:
    """ Transforme un FEN en board virtuel
    par défaut le FEN est le plateau du début de partie, au blanc de jouer"""
    return chess.Board(FEN)


def board_to_fen(board : object) -> str:
    """ Fonction qui transforme un board en FEN """
    return board.fen()

def list_legal_moves(board : object) -> list :
    """ Fonction qui retourne la liste des moves possibles au format 'Move.from_uci('h8g8')'"""
    return list(board.legal_moves)
