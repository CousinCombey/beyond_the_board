import pandas as pd

def create_new_df(df):

    """
    This function takes a DataFrame containing FEN strings and Stockfish scores for different stages of a chess game
    and transforms it into a new DataFrame with two columns: "FEN" and "Stockfish".
    """

    stages = ["opening", "early", "developing", "pre_mid", "midgame", "post_mid", "transition", "late", "pre_end", "endgame"]


    new_df = pd.DataFrame(columns = ["FEN", "Stockfish"])

    # Iterate through each stage and concatenate the FEN strings and Stockfish scores
    for stage in stages:

        white = df[f"fen_{stage}_white"].reset_index(drop=True)

        stockfish = df[f"stockfish_{stage}"].reset_index(drop=True)

        new_df_stage = pd.DataFrame({"FEN": white, "Stockfish": stockfish})

        new_df = pd.concat([new_df, new_df_stage], ignore_index=True)

    return new_df


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('chess_data.csv')  # Replace with your actual data source
    new_df = create_new_df(df)
    print(new_df.head())
