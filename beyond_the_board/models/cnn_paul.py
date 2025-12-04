from beyond_the_board.tensor.main import fen_to_tensor_8_8_12
from keras import Sequential, Input, layers
from sklearn.model_selection import train_test_split
import numpy as np
from keras.callbacks import EarlyStopping

def initialize_model_cnn():
    model_cnn = Sequential()
    model_cnn.add(Input(shape=(8, 8, 12)))
    model_cnn.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model_cnn.add(layers.MaxPooling2D((2, 2)))
    model_cnn.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model_cnn.add(layers.MaxPooling2D((2, 2)))
    model_cnn.add(layers.Flatten())
    model_cnn.add(layers.Dense(64, activation='relu'))
    model_cnn.add(layers.Dense(32, activation='relu'))
    model_cnn.add(layers.Dense(1))
    model_cnn.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model_cnn

def clean_input_model(X,y):

    #train test split
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42)

    #transform into tensors
    X_train_tensors = X_train["FEN"].apply(fen_to_tensor_8_8_12)
    X_test_tensors = X_test["FEN"].apply(fen_to_tensor_8_8_12)

    #refit arrays to be accepted by model
    X_train_arr = np.stack(X_train_tensors.values).astype('float32')
    X_test_arr  = np.stack(X_test_tensors.values).astype('float32')
    y_train_arr = y_train.values.astype('float32')
    y_test_arr  = y_test.values.astype('float32')

    return X_train_arr, X_test_arr, y_train_arr, y_test_arr

def clean_input_predict(fen:str):
    new_fen = 'r1b1k1nN/p1pp3p/1p3p1b/1B4p1/4P3/8/PPPP1PPP/RNBQK2R b KQq - 0 8'
    new_fen_tensor = fen_to_tensor_8_8_12(new_fen)
    new_fen_tensor_arr = np.expand_dims(new_fen_tensor.astype('float32'), axis=0)
    return new_fen_tensor_arr

def train_model(model, X, y):
    X_train_arr, X_test_arr, y_train_arr, y_test_arr = clean_input_model(X,y)
    es = EarlyStopping(patience=20, restore_best_weights=True)
    model.fit(X_train_arr, y_train_arr, validation_split=(0.2), epochs=10000, callbacks=[es], batch_size=32)
    return model

def model_predict(model, fen) :
    fen_tensor = fen_to_tensor_8_8_12(fen)
    fen_tensor_arr = np.expand_dims(fen_tensor.astype('float32'), axis=0)
    prediction = model.predict(fen_tensor_arr)
    return prediction
