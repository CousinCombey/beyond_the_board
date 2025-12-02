import pandas as pd

def create_new_df(df):

    """
    This function takes a DataFrame containing FEN strings and Stockfish scores for different stages of a chess game
    and transforms it into a new DataFrame with two columns: "FEN" and "Stockfish".

    IMPORTANT: Only uses fen_X_white positions with their corresponding stockfish_X scores.
    The fen_X_black positions are one move earlier and should NOT reuse the same stockfish score.
    """

    stages = ["opening", "early", "developing", "pre_mid", "midgame", "post_mid", "transition", "late", "pre_end", "endgame"]


    new_df = pd.DataFrame(columns = ["fen", "stockfish_score"])

    # Iterate through each stage and use ONLY the white FEN positions
    # (The black positions are one move earlier and have different evaluations)
    for stage in stages:

        white_fens = df[f"fen_{stage}_white"].reset_index(drop=True)
        stockfish_scores = df[f"stockfish_{stage}"].reset_index(drop=True)

        new_df_stage = pd.DataFrame({
            "fen": white_fens,
            "stockfish_score": stockfish_scores
        })

        new_df = pd.concat([new_df, new_df_stage], ignore_index=True)

    return new_df


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('chess_data.csv')  # Replace with your actual data source
    new_df = create_new_df(df)
    print(new_df.head())
