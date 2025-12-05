from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import chess
from beyond_the_board.tensor.board_virtuel import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
def predict_model(fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", move):

    chess.Board(fen) # Afficher le board virtuel

    move = 

    """ 
    La fonction prend un Fen, sauf que on peut avoir où le joueur est au milieu et veut commencer la prédiction par son FEN actuel.

    A partir d'un FEN qui est le fen du Board actuel, on prédit le Stockfish actuel, on récupère les moves légaux
    avec python-chess(library), ces moves sont tyransformés en FEN et on refait une prédiction du stockfish de chacun des ces FEN
    ensuyite on fait un tri par ordre décroissant des stcokfish.

    return :
    Stockfish actuel : float
    legal moves : List[(),()...()] ==> list de tuples (move, stockfish) 

    """



@app.get("/")
def root():
    return {"message": "Chess API"}
