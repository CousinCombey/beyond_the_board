from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np

def reshape(X):
    """Fonction qui sert a reshape la matrice 8x8 pour pouvoir l'entrainer dans le model"""
    X_old = np.stack(X.values)
    X = X_old.reshape((X_old.shape[0], -1))

def initial_models(X, y):
    """Premier model crée, avec train test split et early stopping.
    Le model contient 5 neuronnes, 4 activation relu et la derniere une linear.
    Le model es compile en adam, mae loss, et mae metrics"""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    es = EarlyStopping(patience=10, restore_best_weights=True)

    model = Sequential([
    layers.Input(X_train.shape[1:]),
    layers.Dense(8, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mae", metrics=["mae"])
    model.fit(X_train, y_train,
            batch_size=16,
            epochs=100,
            validation_split=0.2,
            callbacks=[es])

    return model, X_test, y_test
