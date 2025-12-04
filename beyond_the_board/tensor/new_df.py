import pandas as pd

def create_new_df_white(df):

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


def create_new_df_black(df):

    """
    This function takes a DataFrame containing FEN strings and Stockfish scores for different stages of a chess game
    and transforms it into a new DataFrame with two columns: "FEN" and "Stockfish".
    """

    stages = ["opening", "early", "developing", "pre_mid", "midgame", "post_mid", "transition", "late", "pre_end", "endgame"]


    new_df = pd.DataFrame(columns = ["FEN", "Stockfish"])

    # Iterate through each stage and concatenate the FEN strings and Stockfish scores
    for stage in stages:

        black = df[f"fen_{stage}_black"].reset_index(drop=True)

        stockfish = df[f"stockfish_{stage}"].reset_index(drop=True)

        new_df_stage = pd.DataFrame({"FEN": black, "Stockfish": stockfish})

        new_df = pd.concat([new_df, new_df_stage], ignore_index=True)

    return new_df

def create_new_df_all(df):

    """
    This function takes a DataFrame containing FEN strings and Stockfish scores for different stages of a chess game
    and transforms it into a new DataFrame with two columns: "FEN" and "Stockfish".
    """

    stages = ["opening", "early", "developing", "pre_mid", "midgame", "post_mid", "transition", "late", "pre_end", "endgame"]


    new_df = pd.DataFrame(columns = ["FEN", "Stockfish"])

    # Iterate through each stage and concatenate the FEN strings and Stockfish scores
    for stage in stages:

        white = df[f"fen_{stage}_white"].reset_index(drop=True)
        black = df[f"fen_{stage}_black"].reset_index(drop=True)
        all_ = pd.concat([white, black], ignore_index=True)

        stockfish = df[f"stockfish_{stage}"].reset_index(drop=True)
        stockfish = pd.concat([stockfish, stockfish])

        new_df_stage = pd.DataFrame({"FEN": black, "Stockfish": stockfish})

        new_df = pd.concat([new_df, new_df_stage], ignore_index=True)

    return new_df


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('chess_data.csv')  # Replace with your actual data source
    new_df = create_new_df_white(df)
    new_df = create_new_df_black(df)
    new_df = create_new_df_all(df)
    print(new_df.head())
