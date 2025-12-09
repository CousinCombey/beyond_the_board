from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import chess
import numpy as np
from beyond_the_board.tensor.board_virtuel import *
from beyond_the_board.tensor.main import fen_to_tensor_8_8_12
from beyond_the_board.ml_logic.chess_pipeline import for_model_predict
from google.cloud import storage
from beyond_the_board.models.load_upload_model import upload_model, load_model
from beyond_the_board.models.cnn_john_full import model_predict
from beyond_the_board.models.pipe2.step3 import predict_all_moves_with_eval_v2
from beyond_the_board.tensor.enhanced_metadata import extract_all_features, get_feature_names


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bucket_name = "beyond_the_board"

# Define source blobs for each model (change these to your actual model names in GCS)
model_configs = {
    "model1": {
        "source_blob": "Trained Models/coach_model.keras",
        "destination": "/beyond_the_board/outputs/coach_model.keras"
    },
    "model2": {
        "source_blob": "Trained Models/train_200k_mate.keras",
        "destination": "/beyond_the_board/outputs/train_200k_mate.keras"
    },
    "model3": {
        "source_blob": "Trained Models/train_200k_mate_full.keras",
        "destination": "/beyond_the_board/outputs/train_200k_mate_full.keras"
    }
}

# Load all 3 models at startup
app.state.model1 = load_model(bucket_name, model_configs["model1"]["source_blob"], model_configs["model1"]["destination"])
app.state.model2 = load_model(bucket_name, model_configs["model2"]["source_blob"], model_configs["model2"]["destination"])
app.state.model3 = load_model(bucket_name, model_configs["model3"]["source_blob"], model_configs["model3"]["destination"])


def detect_model_inputs(model):
    """
    Detect how many inputs the model expects and their shapes.
    Returns: (num_inputs, metadata_features)
    """
    num_inputs = len(model.inputs)
    metadata_features = None

    if num_inputs == 2:
        # Second input is metadata - get its expected size
        metadata_shape = model.inputs[1].shape
        if len(metadata_shape) >= 2:
            metadata_features = int(metadata_shape[-1])

    return num_inputs, metadata_features


def prepare_model_inputs(fen: str, num_inputs: int, metadata_features: int = None):
    """
    Prepare inputs based on model requirements.
    """
    # Always prepare board tensor
    board_tensor = fen_to_tensor_8_8_12(fen)
    board_tensor = np.expand_dims(board_tensor, axis=0).astype('float32')

    if num_inputs == 1:
        # Model expects only board tensor
        return [board_tensor]

    elif num_inputs == 2:
        # Model expects board + metadata
        if metadata_features == 40:
            # Use enhanced metadata (40 features)
            features = extract_all_features(fen)
            feature_names = get_feature_names()
            metadata_vec = np.array([[features[name] for name in feature_names]], dtype='float32')
        else:
            # Use old metadata (15 features)
            metadata_vec = for_model_predict(fen)
            metadata_vec = np.array([metadata_vec], dtype='float32')

        return [board_tensor, metadata_vec]

    else:
        raise ValueError(f"Unsupported number of inputs: {num_inputs}")


def predict_all_moves_format(model, fen: str):
    """
    Helper function that returns the standard format for all endpoints.
    Automatically detects model input requirements.
    Returns: {current_eval, to_move, moves: []}
    """
    board = chess.Board(fen)

    # Detect model input requirements
    num_inputs, metadata_features = detect_model_inputs(model)

    # Get current position evaluation
    model_inputs = prepare_model_inputs(fen, num_inputs, metadata_features)
    prediction = model.predict(model_inputs, verbose=0)

    # Handle multi-output models
    if isinstance(prediction, list):
        current_eval = float(prediction[0][0][0])
    else:
        current_eval = float(prediction[0][0])

    # Determine who's moving
    to_move = 'white' if board.turn == chess.WHITE else 'black'

    # Get all legal moves and their evaluations
    moves_with_eval = []

    for move in board.legal_moves:
        # Apply move to get resulting position
        board_copy = board.copy()
        board_copy.push(move)
        new_fen = board_copy.fen()

        # Predict evaluation of resulting position
        new_model_inputs = prepare_model_inputs(new_fen, num_inputs, metadata_features)
        new_prediction = model.predict(new_model_inputs, verbose=0)

        # Handle multi-output models
        if isinstance(new_prediction, list):
            eval_after = float(new_prediction[0][0][0])
        else:
            eval_after = float(new_prediction[0][0])

        # Calculate improvement from current player's perspective
        if board.turn == chess.WHITE:
            eval_change = eval_after - current_eval
        else:
            eval_change = current_eval - eval_after

        moves_with_eval.append({
            'move_uci': move.uci(),
            'move_san': board.san(move),
            'eval_after': eval_after,
            'eval_change': eval_change,
            'improves': 'True' if eval_change > 0 else 'False'
        })

    # Sort by what's best for current player
    if board.turn == chess.WHITE:
        moves_with_eval.sort(key=lambda x: x['eval_after'], reverse=True)
    else:
        moves_with_eval.sort(key=lambda x: x['eval_after'])

    return {
        'current_eval': current_eval,
        'to_move': to_move,
        'moves': moves_with_eval
    }


@app.get("/Optimus_Prime")
def predict_model(fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
    """
    🎓 The Coach - Your Strategic Chess Advisor

    This model is your experienced chess coach, trained on diverse positions
    to provide balanced, positional evaluations. It excels at:
    - Understanding long-term strategic advantages
    - Evaluating piece placement and pawn structure
    - Providing stable, reliable position assessments

    Best for: General position analysis and strategic planning
    Model: coach_model.keras (40 enhanced features)
    """
    return predict_all_moves_format(app.state.model1, fen)


@app.get("/Shallow_Blue")
def predict_moves(fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
    """
    ⚔️ The Tactician - Checkmate Hunter

    Trained on 200,000 games with focus on checkmate patterns, this model
    is your tactical specialist. It shines at:
    - Spotting forcing sequences and mating attacks
    - Recognizing tactical motifs (pins, forks, skewers)
    - Finding the killing blow in sharp positions

    Best for: Tactical puzzles and attacking positions
    Model: train_200k_mate.keras (mate-focused training)
    """
    return predict_all_moves_format(app.state.model2, fen)


@app.get("/Big_Brother")
def predict_more(fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
    """
    🚀 The Mastermind - Advanced AI Transformer

    Our most sophisticated model using transformer architecture for
    deep pattern recognition. It offers:
    - State-of-the-art neural network evaluation
    - Complex position understanding through attention mechanisms
    - Cutting-edge chess AI insights

    Best for: Complex positions requiring deep analysis
    Model: train_200k_mate_full.keras (mate-focused training)
    """
    return predict_all_moves_format(app.state.model3, fen)


@app.get("/")
def root():
    return {"message": "We are beyond the board"}
