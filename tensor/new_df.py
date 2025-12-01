import pandas as pd

def create_new_df(df):

    """
    Create a new dataframe with FEN strings and Stockfish scores for both white and black turns.
    Args:
        df (pd.DataFrame): Original dataframe containing FEN strings and Stockfish scores for different game stages.
    Returns:
        pd.DataFrame: New dataframe with columns 'fen' and 'stockfish_score' for both white and black turns.
    """
    # Define the game stages
    stages = ['opening', 'early', 'developing', 'pre_mid', 'midgame',
            'post_mid', 'transition', 'late', 'pre_end', 'endgame']
    # Initialize lists to hold dataframes
    dfs_white = []
    dfs_black = []
    # Process each stage
    for stage in stages:
        # Get the relevant columns for the current stage
        fen_white_col = f'fen_{stage}_white'
        fen_black_col = f'fen_{stage}_black'
        stockfish_col = f'stockfish_{stage}'
        # Create dataframe for white's turn positions
        df_white = df[[fen_white_col, stockfish_col]].copy()
        df_white = df_white.rename(columns={
            fen_white_col: 'fen',
            stockfish_col: 'stockfish_score'
        })
        # Add the stage information
        df_white['stage'] = stage

        # Filter out rows where FEN is NaN
        df_white = df_white[df_white['fen'].notna()].copy()
        dfs_white.append(df_white)
        # Create dataframe for black's turn positions
        df_black = df[[fen_black_col, stockfish_col]].copy()
        df_black = df_black.rename(columns={
            fen_black_col: 'fen',
            stockfish_col: 'stockfish_score'
        })
        # Add the stage information
        df_black['stage'] = stage
        # Invert the Stockfish score for black's turn
        df_black['stockfish_score'] = -df_black['stockfish_score']
        # Filter out rows where FEN is NaN
        df_black = df_black[df_black['fen'].notna()].copy()
        dfs_black.append(df_black)
    # Concatenate all dataframes
    fen_stockfish_df = pd.concat(dfs_white + dfs_black, ignore_index=True)
    # Reorder columns
    col_order = ['fen','stockfish_score']
    fen_stockfish_df = fen_stockfish_df[col_order]
    df_new = pd.DataFrame(fen_stockfish_df, columns=fen_stockfish_df.columns)
    df_new.head(10)

    return df_new

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('chess_data.csv')  # Replace with your actual data source
    new_df = create_new_df(df)
    print(new_df.head())
