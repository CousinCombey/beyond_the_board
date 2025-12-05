import pandas as pd
import numpy as np

from keras import Model, Input, layers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from beyond_the_board.models.cnn_paul import fen_to_tensor_8_8_12
from beyond_the_board.tensor.new_df_merged import *

from beyond_the_board.models.cnn_john_full import *
from beyond_the_board.params import *

def read_csv(url):
    """Fonction qui permet de téléchargé le csv"""

    df = pd.read_csv(url)

    return df

def create_input1(df):

    """Fonction qui permet de crée a partir du dataset de base
    un dataset nettoyer avec fen blanc et fen noir et toutes les
    nouvelles colonnes"""

    df = create_new_df_all_merged(df)


    return df

def sample_df(df):

    """Fonction qui sample 200 000 lignes du data set de base pour
    entrainer le model"""

    df = df.sample(200000)


    return df
def data_and_pross(url:str):
    """ rend un df sans le meta data mais bien trié"""
    df = read_csv(url)
    df = create_input1(df)
    df = sample_df(df)

    return df


def pipeline(df):
    """Fonction qui permet tout automatisé et de prédire le prochain coup"""
    meta_data = drop_columns_merged(df)
    meta_data = meta_data.columns.to_list()

    trained_model, history = complete_workflow_example_full(df, "FEN", "Stockfish", meta_data)

    return trained_model, history

def complete_pipeline(url:str):
    """pour l'utiliser : trained_model, history = complete_pipeline('https://storage.googleapis.com/beyond_the_board/Final%20Dataset/games_merged_clean.csv')"""
    df = data_and_pross(url)

    trained_model, history = pipeline(df)

    return trained_model, history
