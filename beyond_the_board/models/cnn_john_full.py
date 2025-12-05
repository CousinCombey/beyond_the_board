"""
CNN Model for Chess Position Evaluation - FULL VERSION WITH METADATA FEATURES

This module provides a complete pipeline for training and using a Convolutional Neural Network
to evaluate chess positions from FEN strings, INCLUDING 15 metadata features.

DIFFERENCE FROM cnn_john.py:
    - input_2 now uses 15 REAL features (not zeros):
        * turn_enc (1 feature)
        * castling rights (4 features)
        * en passant one-hot (8 features)
        * halfmove_norm (1 feature)
        * fullmove_norm (1 feature)

Configuration:
    All hyperparameters are centralized in params.py and loaded from .env file:
    - DATA_SIZE: Amount of data to use (1k, 10k, 100k, 200k, all)
    - RANDOM_SEED: Random seed for reproducibility
    - TEST_SIZE: Fraction of data for testing
    - VALIDATION_SIZE: Fraction of training data for validation
    - BATCH_SIZE: Batch size for training
    - MAX_EPOCHS: Maximum number of training epochs
    - EARLY_STOPPING_PATIENCE: Epochs to wait before stopping if no improvement

Usage:
    from beyond_the_board.models.cnn_john_full import complete_workflow_example
    import pandas as pd

    df = pd.read_csv('your_data.csv')  # Must contain FEN column + 15 metadata columns
    trained_model = complete_workflow_example(df)
"""

from beyond_the_board.tensor.main import fen_to_tensor_8_8_12
from beyond_the_board.params import (
    RANDOM_SEED,
    TEST_SIZE,
    VALIDATION_SIZE,
    BATCH_SIZE,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    DATA_SIZE,
    get_data_size_limit
)
from keras import Model, Input, layers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np

# ------------- Configuration Summary ----------------

def print_model_config():
    """
    Print current model configuration from params.

    Useful for debugging and tracking experiment settings.
    """
    print("\n" + "=" * 60)
    print("MODEL CONFIGURATION (from .env)")
    print("=" * 60)
    print(f"DATA_SIZE:                {DATA_SIZE}")
    print(f"RANDOM_SEED:              {RANDOM_SEED}")
    print(f"TEST_SIZE:                {TEST_SIZE}")
    print(f"VALIDATION_SIZE:          {VALIDATION_SIZE}")
    print(f"BATCH_SIZE:               {BATCH_SIZE}")
    print(f"MAX_EPOCHS:               {MAX_EPOCHS}")
    print(f"EARLY_STOPPING_PATIENCE:  {EARLY_STOPPING_PATIENCE}")
    print("=" * 60 + "\n")

# ------------- Initialize Model ----------------

def initialize_model_cnn_john_full():
    """
    Initialize the CNN model with FULL metadata features (15 instead of 8 dummy zeros).

    Architecture:
    - Input 1: (None, 8, 8, 12) chess board tensor (12 channels for 12 piece types)
    - Input 2: (None, 15) metadata features from FEN (turn, castling, en passant, halfmove, fullmove)
    - Multiple convolutional blocks with batch normalization
    - Skip connections via concatenation layers
    - Dense layers for final evaluation
    - Output: Single value (position evaluation)

    Returns:
        keras.Model: Compiled CNN model ready for training
    """
    # Main input: chess board tensor (8x8x12)
    input_1 = Input(shape=(8, 8, 12), name='input_1')

    # First residual block
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2d')(input_1)
    x = layers.BatchNormalization(name='batch_normalization')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2d_1')(x)
    x = layers.BatchNormalization(name='batch_normalization_1')(x)
    skip_1 = layers.Concatenate(name='concatenate')([input_1, x])

    # Second residual block
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2d_2')(skip_1)
    x = layers.BatchNormalization(name='batch_normalization_2')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2d_3')(x)
    x = layers.BatchNormalization(name='batch_normalization_3')(x)
    skip_2 = layers.Concatenate(name='concatenate_1')([skip_1, x])

    # Third residual block
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2d_4')(skip_2)
    x = layers.BatchNormalization(name='batch_normalization_4')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2d_5')(x)
    x = layers.BatchNormalization(name='batch_normalization_5')(x)
    skip_3 = layers.Concatenate(name='concatenate_2')([skip_2, x])

    # Flatten convolutional output
    flatten = layers.Flatten(name='flatten')(skip_3)

    # Secondary input: 15 metadata features (CHANGED FROM 8!)
    input_2 = Input(shape=(15,), name='input_2')
    dense_input_2 = layers.Dense(15, activation='relu', name='dense')(input_2)

    # Combine both inputs
    combined = layers.Concatenate(name='concatenate_3')([flatten, dense_input_2])

    # Final dense layers
    x = layers.Dense(4096, activation='relu', name='dense_1')(combined)
    output = layers.Dense(1, name='output')(x)

    # Create model
    model = Model(
        inputs=[input_1, input_2],
        outputs=output,
        name='chess_cnn_full'
    )

    # Compile model
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model

# ------------- Prepare Data ----------------

def prepare_data(df, fen_column, target_column, metadata_columns=None, limit_data=True):
    """
    Prepare features (X) and target (y) from DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with chess positions and evaluations
        fen_column (str): Name of column containing FEN strings (default: 'FEN')
        target_column (str): Name of column containing evaluation scores (default: 'Evaluation')
        metadata_columns (list): List of metadata column names to include in X (optional)
        limit_data (bool): Whether to apply DATA_SIZE limit from params (default: True)

    Returns:
        tuple: (X, y) - Features DataFrame and target Series
    """
    # Apply data size limit from params if requested
    if limit_data:
        data_limit = get_data_size_limit()
        if data_limit is not None:
            df = df.head(data_limit)
            print(f"Data limited to {data_limit} rows (DATA_SIZE={DATA_SIZE})")

    # Select columns for X
    if metadata_columns is not None:
        X = df[[fen_column] + metadata_columns]
    else:
        X = df[[fen_column]]

    y = df[target_column]

    return X, y

# ------------- Create Train-Test Split ----------------

def create_train_test_split(X, y, test_size=None, random_state=None):
    """
    Split data into training and testing sets using params from .env

    Args:
        X: Features
        y: Target values
        test_size: Fraction for test set (defaults to TEST_SIZE from params)
        random_state: Random seed (defaults to RANDOM_SEED from params)

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    if test_size is None:
        test_size = TEST_SIZE
    if random_state is None:
        random_state = RANDOM_SEED

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

# ------------- Convert FEN to Tensors ----------------

def convert_fen_to_tensors(X_train, X_test, fen_column):
    """
    Convert FEN strings to tensor representations.

    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        fen_column (str): Name of FEN column (default: 'FEN')

    Returns:
        tuple: (X_train_tensors, X_test_tensors) - Series of tensors
    """
    X_train_tensors = X_train[fen_column].apply(fen_to_tensor_8_8_12)
    X_test_tensors = X_test[fen_column].apply(fen_to_tensor_8_8_12)

    return X_train_tensors, X_test_tensors

# ------------- Prepare Arrays ----------------

def prepare_arrays_for_model(X_train_tensors, X_test_tensors, y_train, y_test):
    """
    Convert pandas Series to numpy arrays suitable for model input.

    Stacks individual tensors into batched arrays and ensures proper data types.

    Args:
        X_train_tensors (pd.Series): Training tensors
        X_test_tensors (pd.Series): Testing tensors
        y_train (pd.Series): Training labels
        y_test (pd.Series): Testing labels

    Returns:
        tuple: (X_train_arr, X_test_arr, y_train_arr, y_test_arr) as numpy arrays
    """
    X_train_arr = np.stack(X_train_tensors.values).astype('float32')
    X_test_arr = np.stack(X_test_tensors.values).astype('float32')

    y_train_arr = y_train.values.astype('float32')
    y_test_arr = y_test.values.astype('float32')

    return X_train_arr, X_test_arr, y_train_arr, y_test_arr

# ------------- Create Secondary Input ----------------

def create_secondary_input(X_train, X_test, metadata_columns):
    """
    Create secondary input features for the model using REAL metadata from DataFrame.

    The 15 metadata columns are:
    - turn_enc (1)
    - white_king_castling, white_queen_castling, black_king_castling, black_queen_castling (4)
    - ep_a, ep_b, ep_c, ep_d, ep_e, ep_f, ep_g, ep_h (8)
    - halfmove_norm, fullmove_norm (2)

    Args:
        X_train (pd.DataFrame): Training features with metadata columns
        X_test (pd.DataFrame): Testing features with metadata columns
        metadata_columns (list): List of 15 column names containing metadata

    Returns:
        tuple: (X_train_input2, X_test_input2) - secondary input arrays of shape (n, 15)
    """

    X_train_input2 = X_train[metadata_columns].values.astype('float32')
    X_test_input2 = X_test[metadata_columns].values.astype('float32')

    return X_train_input2, X_test_input2

# ------------- Clean Input Model ----------------

def clean_input_model(X, y, fen_column, metadata_columns):
    """
    Complete data preprocessing pipeline for model training with FULL metadata features.

    Orchestrates all preprocessing steps:
    1. Train-test split
    2. FEN to tensor conversion
    3. Array preparation
    4. Secondary input creation from metadata columns

    Args:
        X (pd.DataFrame): Features (FEN strings + metadata columns)
        y (pd.Series): Target values (evaluation scores)
        fen_column (str): Name of FEN column in X
        metadata_columns (list): List of 15 metadata column names

    Returns:
        tuple: (X_train_inputs, X_test_inputs, y_train_arr, y_test_arr)
               where X_train_inputs and X_test_inputs are lists of [board_tensor, secondary_features]
    """
    # Step 1: Split data into train and test sets
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    # Step 2: Convert FEN strings to tensors
    X_train_tensors, X_test_tensors = convert_fen_to_tensors(X_train, X_test, fen_column)

    # Step 3: Prepare arrays for model input
    X_train_arr, X_test_arr, y_train_arr, y_test_arr = prepare_arrays_for_model(
        X_train_tensors, X_test_tensors, y_train, y_test
    )

    # Step 4: Create secondary input features from metadata columns
    X_train_input2, X_test_input2 = create_secondary_input(X_train, X_test, metadata_columns)

    # Combine inputs for multi-input model
    X_train_inputs = [X_train_arr, X_train_input2]
    X_test_inputs = [X_test_arr, X_test_input2]

    return X_train_inputs, X_test_inputs, y_train_arr, y_test_arr

# ------------- Train Model ----------------

def train_model(model, X, y, fen_column, metadata_columns, validation_split=None, epochs=None, batch_size=None, patience=None):
    """
    Train the CNN model on chess position data using params from .env

    Includes early stopping to prevent overfitting by monitoring validation loss.

    Args:
        model (keras.Model): Initialized CNN model
        X (pd.DataFrame): Features (FEN strings + metadata columns)
        y (pd.Series): Target values (evaluation scores)
        fen_column (str): Name of FEN column in X
        metadata_columns (list): List of 15 metadata column names
        validation_split (float): Fraction of training data to use for validation (defaults to VALIDATION_SIZE from params)
        epochs (int): Maximum number of training epochs (defaults to MAX_EPOCHS from params)
        batch_size (int): Number of samples per gradient update (defaults to BATCH_SIZE from params)
        patience (int): Number of epochs with no improvement before stopping (defaults to EARLY_STOPPING_PATIENCE from params)

    Returns:
        tuple: (trained_model, history) - trained model and training history
    """
    # Use params from .env if not explicitly provided
    if validation_split is None:
        validation_split = VALIDATION_SIZE
    if epochs is None:
        epochs = MAX_EPOCHS
    if batch_size is None:
        batch_size = BATCH_SIZE
    if patience is None:
        patience = EARLY_STOPPING_PATIENCE
    # Prepare data for training
    X_train_inputs, _, y_train_arr, _ = clean_input_model(X, y, fen_column, metadata_columns)

    # Early stopping callback: stops training when validation loss stops improving
    # restore_best_weights: restores model weights from epoch with best validation loss
    es = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    # Train the model
    # validation_split: automatically splits training data for validation
    history = model.fit(
        X_train_inputs,
        y_train_arr,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )

    return model, history

# ------------- Evaluate Model ----------------

def evaluate_model(model, X, y, fen_column, metadata_columns):
    """
    Evaluate model performance on test data.

    Args:
        model (keras.Model): Trained model
        X (pd.DataFrame): Features (FEN strings + metadata columns)
        y (pd.Series): Target values (evaluation scores)
        fen_column (str): Name of FEN column in X
        metadata_columns (list): List of 15 metadata column names

    Returns:
        dict: Dictionary containing evaluation metrics (loss, mae)
    """
    # Prepare test data
    _, X_test_inputs, _, y_test_arr = clean_input_model(X, y, fen_column, metadata_columns)

    # Evaluate on test set
    test_loss, test_mae = model.evaluate(X_test_inputs, y_test_arr, verbose=0)

    # Store metrics in dictionary
    metrics = {
        'test_loss': test_loss,
        'test_mae': test_mae
    }

    print(f"\nTest Set Evaluation:")
    print(f"  Loss (MSE): {test_loss:.4f}")
    print(f"  MAE: {test_mae:.4f}")

    return metrics

# ------------- Single prediction ----------------

def prepare_single_prediction(fen, metadata_array):
    """
    Prepare a single FEN string for model prediction.

    Converts FEN to tensor and creates the required input format for the model.

    Args:
        fen (str): FEN string representing chess position
        metadata_array (np.ndarray or list): Array of 15 metadata features

    Returns:
        list: [board_tensor, secondary_features] ready for model prediction
    """
    # Convert FEN to tensor
    fen_tensor = fen_to_tensor_8_8_12(fen)

    # Add batch dimension: (8, 8, 12) -> (1, 8, 8, 12)
    fen_tensor_arr = np.expand_dims(fen_tensor.astype('float32'), axis=0)

    # Create secondary input (15 features)
    secondary_input = np.array([metadata_array], dtype='float32')

    return [fen_tensor_arr, secondary_input]

# ------------- Model prediction ----------------

def model_predict(model, fen, metadata_array):
    """
    Predict evaluation score for a chess position.

    Args:
        model (keras.Model): Trained model
        fen (str): FEN string representing chess position
        metadata_array (np.ndarray or list): Array of 15 metadata features

    Returns:
        float: Predicted evaluation score
    """
    # Prepare input data
    model_inputs = prepare_single_prediction(fen, metadata_array)

    # Make prediction
    prediction = model.predict(model_inputs, verbose=0)

    # Return scalar value (remove batch dimension)
    return prediction[0][0]

# ------------- batch prediction ----------------

def batch_predict(model, fen_list, metadata_arrays):
    """
    Predict evaluation scores for multiple chess positions.

    More efficient than calling model_predict multiple times.

    Args:
        model (keras.Model): Trained model
        fen_list (list): List of FEN strings
        metadata_arrays (np.ndarray or list of lists): Array of metadata features, shape (n, 15)

    Returns:
        np.ndarray: Array of predicted evaluation scores
    """
    # Convert all FENs to tensors
    tensors = [fen_to_tensor_8_8_12(fen) for fen in fen_list]

    # Stack into batch array
    batch_arr = np.stack(tensors).astype('float32')

    # Create secondary input from metadata
    secondary_input = np.array(metadata_arrays, dtype='float32')

    # Make batch prediction
    predictions = model.predict([batch_arr, secondary_input], verbose=0)

    # Return flattened predictions
    return predictions.flatten()

# ------------- usage workflow ----------------

def complete_workflow_example_full(df, fen_column, target_column, metadata_columns):
    """
    Complete example workflow from data loading to prediction - FULL VERSION.

    Demonstrates the full pipeline with 15 metadata features:
    1. Configuration display
    2. Data preparation
    3. Model initialization
    4. Training
    5. Evaluation
    6. Prediction

    Args:
        df (pd.DataFrame): Input DataFrame with chess positions, evaluations, and metadata
        fen_column (str): Name of FEN column (default: 'FEN')
        target_column (str): Name of evaluation column (default: 'Stockfish')
        metadata_columns (list): List of 15 metadata column names

    Returns:
        keras.Model: Trained model ready for predictions
    """
    print("=" * 60)
    print("CHESS POSITION EVALUATION - COMPLETE WORKFLOW (FULL)")
    print("=" * 60)

    # Step 0: Display configuration
    print_model_config()

    # Step 1: Prepare X and y from DataFrame (applies DATA_SIZE limit)
    print("\n[1/5] Preparing data...")
    X, y = prepare_data(df, fen_column, target_column, metadata_columns, limit_data=True)
    print(f"  Loaded {len(X)} positions with 15 metadata features")

    # Step 2: Initialize the model
    print("\n[2/5] Initializing model...")
    model = initialize_model_cnn_john_full()
    print(f"  Model created with {model.count_params():,} parameters")
    model.summary()

    # Step 3: Train the model
    print("\n[3/5] Training model...")
    trained_model, history = train_model(model, X, y, fen_column, metadata_columns)
    print(f"  Training completed in {len(history.history['loss'])} epochs")

    # Step 4: Evaluate the model
    print("\n[4/5] Evaluating model...")
    evaluate_model(trained_model, X, y, fen_column, metadata_columns)

    # Step 5: Make predictions
    print("\n[5/5] Testing predictions...")
    sample_fen = X.iloc[0][fen_column]
    sample_metadata = X.iloc[0][metadata_columns].values
    prediction = model_predict(trained_model, sample_fen, sample_metadata)
    actual = y.iloc[0]
    print(f"  Sample prediction:")
    print(f"    FEN: {sample_fen}")
    print(f"    Metadata: {sample_metadata}")
    print(f"    Predicted: {prediction:.2f}")
    print(f"    Actual: {actual:.2f}")
    print(f"    Error: {abs(prediction - actual):.2f}")

    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)

    return trained_model, history
