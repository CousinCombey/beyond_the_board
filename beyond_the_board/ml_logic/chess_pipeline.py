import pandas as pd
import numpy as np

from keras import Model, Input, layers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from beyond_the_board.models.cnn_paul import fen_to_tensor_8_8_12
from beyond_the_board.tensor.new_df import create_new_df_white, drop_columns, create_new_df_black

from beyond_the_board.models.cnn_john import *
from beyond_the_board.params import *
def data_and_pross():

    def read_csv(url):
        """Fonction qui permet de téléchargé le csv"""

        df = pd.read_csv(url)

        return df

    def create_input1(df):

        """Fonction qui permet de crée a partir du dataset de base
        un dataset ou c'est au blanc de jouer, et un ou c'est les noirs"""

        dfw = create_new_df_white(df)
        dfb = create_new_df_black(df)

        return dfw, dfb

    def sample_df(dfw, dfb):

        """Fonction qui sample 200 000 lignes du data set de base pour
        entrainer le model"""
        dfw = dfw.sample(200000)
        dfb = dfb.sample(200000)


def pipeline():
    """Fonction qui permet tout automatisé et de prédire le prochain coup"""

    pass
