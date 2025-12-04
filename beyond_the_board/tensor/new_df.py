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

    new_df = df_fen_cut(new_df)

    new_df = df_final_cut(new_df)



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

    new_df = df_fen_cut(new_df)

    new_df = df_final_cut(new_df)



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

    new_df = df_fen_cut(new_df)

    new_df = df_final_cut(new_df)



    return new_df

def df_fen_cut(df):

    """Fonction qui crée 6 nouvelles colonnes dans le DataFrame
       avec les 6 paramètre de la FEN à partir de la colonne nommée 'FEN' """
    df['FEN_board'] = df["FEN"].apply(lambda x: x.split(" ")[0])
    df['FEN_player'] = df["FEN"].apply(lambda x: x.split(" ")[1])
    df['FEN_roque'] = df["FEN"].apply(lambda x: x.split(" ")[2])
    df['FEN_passant'] = df["FEN"].apply(lambda x: x.split(" ")[3])
    df['FEN_count_nul'] = df["FEN"].apply(lambda x: x.split(" ")[4])
    df['FEN_coup_complet'] = df["FEN"].apply(lambda x: x.split(" ")[5])

    return df
def df_final_cut(df):



    # Transforme le tour du joueur w or b en chiffre
    df["turn_enc"] = (df["FEN_player"] == "w").astype(int)

    # Transforme els roques en 4 différentes colonnes possibles
    df["white_king_castling"]  = df["FEN_roque"].apply(lambda x: "K" in x).astype(int)
    df["white_queen_castling"] = df["FEN_roque"].apply(lambda x: "Q" in x).astype(int)
    df["black_king_castling"]  = df["FEN_roque"].apply(lambda x: "k" in x).astype(int)
    df["black_queen_castling"] = df["FEN_roque"].apply(lambda x: "q" in x).astype(int)

    # -Transforme les en passant en 8 colonnes possible en fonction de leurs lettre sur l'échiquier
    files = list("abcdefgh")
    for f in files:
        df[f"ep_{f}"] = df["FEN_passant"].apply(lambda x: int(x != "-" and x[0] == f))

    # Cdemi coup normalisé
    df["halfmove_norm"] = df["FEN_count_nul"].astype(int) / 100.0

    # Coup complet (une fois que le blanc, puis le noir on joué) normalisé
    df["fullmove_norm"] = df["FEN_coup_complet"].astype(int) / 200.0

    return df

def drop_columns(df):

    columns_to_drop = [
        "FEN",
        "FEN_board",
        "FEN_player",
        "FEN_roque",
        "FEN_passant",
        "FEN_count_nul",
        "FEN_coup_complet",
        "Stockfish"
    ]
    return df.drop(columns=columns_to_drop)



if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('chess_data.csv')  # Replace with your actual data source
    new_df = create_new_df(df)
    print(new_df.head())
