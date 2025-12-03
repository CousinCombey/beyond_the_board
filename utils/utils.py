import matplotlib.pyplot as plt

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
    







# --- End of visualization functions ---
