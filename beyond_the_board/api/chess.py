from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import chess
from beyond_the_board.tensor.board_virtuel import *
from google.cloud import storage
from beyond_the_board.models.load_upload_model import upload_model, load_model
from beyond_the_board.ml_logic.chess_pipeline import for_model_predict
from beyond_the_board.models.cnn_john_full import model_predict


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bucket_name = "beyond_the_board"
source_blob_name = "Trained Models/train_200k_mate.keras"
destination_file_name = "/beyond_the_board/outputs/train_200k_mate.keras"

app.state.model = load_model(bucket_name, source_blob_name, destination_file_name)
@app.get("/predict")
def predict_model(fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):

    """
    La fonction prend un Fen, sauf que on peut avoir où le joueur est au milieu et veut commencer la prédiction par son FEN actuel.

    A partir d'un FEN qui est le fen du Board actuel, on prédit le Stockfish actuel, on récupère les moves légaux
    avec python-chess(library), ces moves sont tyransformés en FEN et on refait une prédiction du stockfish de chacun des ces FEN
    ensuyite on fait un tri par ordre décroissant des stcokfish.

    return :
    Stockfish actuel : float
    legal moves : List[(),()...()] ==> list de tuples (move, stockfish)

    """

    fen_preprocessed = for_model_predict(fen)

    prediction = model_predict(app.state.model, fen, fen_preprocessed)

        # FIX: convert numpy values to Python native types
    if isinstance(prediction, dict):
        return {k: float(v) for k, v in prediction.items()}

    if isinstance(prediction, (list, tuple)):
        return [float(x) for x in prediction]

    # Single value (numpy.float32)
    return float(prediction)

@app.get("/")
def root():
    return {"message": "We are beyond the board"}
