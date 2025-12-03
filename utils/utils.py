import matplotlib.pyplot as plt
import os
import numpy as np
from beyond_the_board.params import FILES, INDEX_TO_SQUARE, MAX_FEN_LENGTH, MIN_FEN_LENGTH, PIECE_VALUES, RANKS, SQUARE_TO_INDEX

# --- Visualization functions ---

def plot_stockfish_endgame_distribution(df):
    df.plot.hist(column='stockfish_endgame', bins=50, figsize=(10, 5))
    plt.title('Distribution of Stockfish Endgame Evaluations')
    plt.xlabel('Stockfish Endgame Evaluation')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
plt.show()


def plot_checkmate_vs_resign(df):
    normal_games = df[df['termination'] == 'Normal']
    checkmate_count = normal_games['term_checkmate'].sum()
    resign_count = len(normal_games) - checkmate_count

    labels = ['Checkmate', 'Resign']
    counts = [checkmate_count, resign_count]

    plt.figure(figsize=(6, 6))
    plt.bar(labels, counts, color=['green', 'red'])
    plt.title('Checkmate vs Resign in Normal Terminations')
    plt.ylabel('Number of Games')
    plt.show()


def plot_checkmate_white_distribution(df):
    checkmate_games = df[df['term_checkmate'] == 1]
    white_checkmate_count = checkmate_games['checkmate_white'].sum()
    black_checkmate_count = len(checkmate_games) - white_checkmate_count

    labels = ['White Checkmate', 'Black Checkmate']
    counts = [white_checkmate_count, black_checkmate_count]

    plt.figure(figsize=(6, 6))
    plt.bar(labels, counts, color=['gray', 'black'])
    plt.title('Checkmate Distribution by Color')
    plt.ylabel('Number of Checkmates')
    plt.show()

# --- Plot model history ---

# Plot model history function
def plot_model_history(history):
    plt.figure(figsize=(12, 5))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)

    # Plot training & validation MAE values
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mae'], label='Train MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)

    plt.tight_layout()
    plt.show()

# --- Plot learning curve ---

def plot_learning_curve(history, save_path=None):
    """
    Plot the learning curve showing training and validation metrics over epochs.

    Parameters:
    -----------
    history : dict or keras.callbacks.History
        Training history containing metrics. Can be a dict with keys like 'loss', 'val_loss', etc.
        or a Keras History object with a .history attribute.
    save_path : str, optional
        If provided, saves the plot to this path instead of displaying it.

    Example:
    --------
    >>> history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
    >>> plot_learning_curve(history)
    """
    # Handle both dict and Keras History object
    if hasattr(history, 'history'):
        history = history.history

    # Determine available metrics
    metrics = [key for key in history.keys() if not key.startswith('val_')]

    # Calculate number of subplots needed
    n_metrics = len(metrics)

    if n_metrics == 0:
        print("No metrics found in history to plot.")
        return

    # Create subplots
    _, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

    # Ensure axes is always a list
    if n_metrics == 1:
        axes = [axes]

    # Plot each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Plot training metric
        epochs = range(1, len(history[metric]) + 1)
        ax.plot(epochs, history[metric], 'b-', linewidth=2, label=f'Training {metric}')

        # Plot validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(epochs, history[val_metric], 'r-', linewidth=2, label=f'Validation {metric}')

        # Formatting
        ax.set_title(f'{metric.upper()} Learning Curve', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(metric.upper(), fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve saved to {save_path}")
    else:
        plt.show()


# --- End of visualization functions ---

# --- Params Functions ---

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

def load_data(file_path: str):
    """Loads data from a given file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    # Add your data loading logic here
    data = None
    return data
