import os
import numpy as np
from pathlib import Path
# from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent / '.env'
# load_dotenv(dotenv_path=env_path)

##################  VARIABLES  ##################

DATA_SIZE = os.environ.get("DATA_SIZE")
if DATA_SIZE:
    DATA_SIZE = DATA_SIZE.strip("'\"")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
MODEL_OUTPUT_PATH = os.environ.get("MODEL_OUTPUT_PATH", "models/")
RANDOM_SEED = int(os.environ.get("RANDOM_SEED"))
TEST_SIZE = float(os.environ.get("TEST_SIZE"))
VALIDATION_SIZE = float(os.environ.get("VALIDATION_SIZE"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS"))
EARLY_STOPPING_PATIENCE = int(os.environ.get("EARLY_STOPPING_PATIENCE"))
STOCKFISH_EVAL_TIME = float(os.environ.get("STOCKFISH_EVAL_TIME"))
STOCKFISH_MOVE_TIME = float(os.environ.get("STOCKFISH_MOVE_TIME"))
RAW_DATA_PATH = os.environ.get("RAW_DATA_PATH")
CLEAN_DATA_PATH = os.environ.get("CLEAN_DATA_PATH")
DATA_WITH_FEN = os.environ.get("DATA_WITH_FEN")
DATA_WITH_NEW_FEATURES = os.environ.get("DATA_WITH_NEW_FEATURES")
DATA_WITH_PGN = os.environ.get("DATA_WITH_PGN")

################## VALIDATIONS #################

env_valid_options = dict(
    DATA_SIZE=["1k", "10k", "100k", "200k", "all"],
    MODEL_TARGET=["local", "gcs", "mlflow"],
)

def validate_env_value(env, valid_options):
    env_value = os.environ[env]
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")


for env, valid_options in env_valid_options.items():
    validate_env_value(env, valid_options)

##################  CONSTANTS  ##################

PIECES = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
PIECE_TO_INDEX = {piece: index for index, piece in enumerate(PIECES)}
INDEX_TO_PIECE = {index: piece for index, piece in enumerate(PIECES)}
EMPTY_SQUARE_INDEX = 12
INDEX_TO_PIECE[EMPTY_SQUARE_INDEX] = '.'
PIECE_TO_INDEX['.'] = EMPTY_SQUARE_INDEX
BOARD_SIZE = 8
NUM_PIECE_TYPES = len(PIECES) + 1  # Including empty square
MAX_MOVES_PER_GAME = 200
INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
SAMPLE_FEN = "r1b1k1nN/p1pp3p/1p3p1b/1B4p1/4P3/8/PPPP1PPP/RNBQK2R b KQq - 0 8"
CHECKMATE_SCORE = 10000
DRAW_SCORE = 0
MAX_SCORE = 20000
MIN_SCORE = -20000
INFINITY = float('inf')
NEGATIVE_INFINITY = float('-inf')
PIECE_VALUES = {
    'P': 100,
    'N': 320,
    'B': 330,
    'R': 500,
    'Q': 900,
    'K': 20000,
    'p': -100,
    'n': -320,
    'b': -330,
    'r': -500,
    'q': -900,
    'k': -20000
}
FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
RANKS = ['1', '2', '3', '4', '5', '6', '7', '8']
FILE_TO_INDEX = {file: index for index, file in enumerate(FILES)}
RANK_TO_INDEX = {rank: index for index, rank in enumerate(RANKS)}
INDEX_TO_FILE = {index: file for index, file in enumerate(FILES)}
INDEX_TO_RANK = {index: rank for index, rank in enumerate(RANKS)}
CASTLING_RIGHTS = ['K', 'Q', 'k', 'q']
CASTLING_TO_INDEX = {right: index for index, right in enumerate(CASTLING_RIGHTS)}
INDEX_TO_CASTLING = {index: right for index, right in enumerate(CASTLING_RIGHTS)}
EMPTY_CASTLING_INDEX = 4
CASTLING_TO_INDEX['-'] = EMPTY_CASTLING_INDEX
INDEX_TO_CASTLING[EMPTY_CASTLING_INDEX] = '-'
MAX_PLY_DEPTH = 64
DIRECTIONS = {
    'N': (-1, 0),
    'S': (1, 0),
    'E': (0, 1),
    'W': (0, -1),
    'NE': (-1, 1),
    'NW': (-1, -1),
    'SE': (1, 1),
    'SW': (1, -1)
}
KNIGHT_MOVES = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
KING_MOVES = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
PROMOTION_PIECES = ['Q', 'R', 'B', 'N']
MAX_FEN_LENGTH = 100
MIN_FEN_LENGTH = 20
SQUARES = [f + r for r in RANKS for f in FILES]
SQUARE_TO_INDEX = {square: index for index, square in enumerate(SQUARES)}
INDEX_TO_SQUARE = {index: square for index, square in enumerate(SQUARES)}

##################  FUNCTIONS  ##################

def get_data_size_limit() -> int:
    """
    Convert DATA_SIZE string to integer limit for data loading.

    Returns:
        int: Number of rows to load, or None for all data
    """
    size_map = {
        "1k": 1000,
        "10k": 10000,
        "100k": 100000,
        "200k": 200000,
        "all": None
    }
    return size_map.get(DATA_SIZE, None)

def get_piece_value(piece: str) -> int:
    """Returns the material value of a given piece."""
    return PIECE_VALUES.get(piece, 0)

def is_valid_square(square: str) -> bool:
    """Checks if a given square notation is valid."""
    if len(square) != 2:
        return False
    file, rank = square[0], square[1]
    return file in FILES and rank in RANKS

def index_to_square(index: int) -> str:
    """Converts a board index to standard chess square notation."""
    return INDEX_TO_SQUARE.get(index, None)

def square_to_index(square: str) -> int:
    """Converts standard chess square notation to a board index."""
    return SQUARE_TO_INDEX.get(square, -1)

def is_valid_fen(fen: str) -> bool:
    """Basic validation to check if a FEN string is of plausible length."""
    return MIN_FEN_LENGTH <= len(fen) <= MAX_FEN_LENGTH

def reset_model_output_path(path: str):
    """Resets the model output directory."""
    global MODEL_OUTPUT_PATH
    MODEL_OUTPUT_PATH = path
    if not os.path.exists(MODEL_OUTPUT_PATH):
        os.makedirs(MODEL_OUTPUT_PATH)
       # Optionally, you can add code here to reset any existing model files
    else:
        for filename in os.listdir(MODEL_OUTPUT_PATH):
            file_path = os.path.join(MODEL_OUTPUT_PATH, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def set_random_seed(seed: int):
    """Sets the random seed for reproducibility."""
    global RANDOM_SEED
    RANDOM_SEED = seed
    np.random.seed(RANDOM_SEED)
    import random
    random.seed(RANDOM_SEED)

set_random_seed(RANDOM_SEED)

##################  END OF FILE  ##################
