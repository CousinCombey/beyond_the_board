"""
CNN Model for Chess Position Evaluation - REGULARIZED VERSION

This module addresses overfitting issues by adding:
1. Dropout layers for regularization
2. L2 weight regularization
3. Reduced model complexity option
4. Data augmentation capabilities
5. Learning rate scheduling
6. Model comparison utilities

Based on cnn_john_full.py but with additional regularization techniques.

Usage:
    from beyond_the_board.models.cnn_john_regularized import complete_workflow_regularized
    import pandas as pd

    df = pd.read_csv('your_data.csv')
    trained_model = complete_workflow_regularized(df)
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
from keras import Model, Input, layers, regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# ------------- Regularized Model Architectures ----------------

def initialize_model_with_dropout(dropout_rate=0.6, l2_reg=0):
    """
    Initialize CNN model with dropout and L2 regularization to prevent overfitting.

    Key differences from original:
    - Dropout layers after each conv block
    - L2 regularization on Conv2D and Dense layers
    - Dropout before final dense layer

    Args:
        dropout_rate (float): Dropout rate (0.0 to 1.0). Default 0.3
        l2_reg (float): L2 regularization strength. Default 0.001

    Returns:
        keras.Model: Compiled CNN model with regularization
    """
    # Main input: chess board tensor (8x8x12)
    input_1 = Input(shape=(8, 8, 12), name='input_1')

    # First residual block with regularization
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(l2_reg), name='conv2d')(input_1)
    x = layers.BatchNormalization(name='batch_normalization')(x)
    x = layers.Dropout(dropout_rate, name='dropout_1')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(l2_reg), name='conv2d_1')(x)
    x = layers.BatchNormalization(name='batch_normalization_1')(x)
    x = layers.Dropout(dropout_rate, name='dropout_2')(x)
    skip_1 = layers.Concatenate(name='concatenate')([input_1, x])

    # Second residual block with regularization
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(l2_reg), name='conv2d_2')(skip_1)
    x = layers.BatchNormalization(name='batch_normalization_2')(x)
    x = layers.Dropout(dropout_rate, name='dropout_3')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(l2_reg), name='conv2d_3')(x)
    x = layers.BatchNormalization(name='batch_normalization_3')(x)
    x = layers.Dropout(dropout_rate, name='dropout_4')(x)
    skip_2 = layers.Concatenate(name='concatenate_1')([skip_1, x])

    # Third residual block with regularization
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(l2_reg), name='conv2d_4')(skip_2)
    x = layers.BatchNormalization(name='batch_normalization_4')(x)
    x = layers.Dropout(dropout_rate, name='dropout_5')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(l2_reg), name='conv2d_5')(x)
    x = layers.BatchNormalization(name='batch_normalization_5')(x)
    x = layers.Dropout(dropout_rate, name='dropout_6')(x)
    skip_3 = layers.Concatenate(name='concatenate_2')([skip_2, x])

    # Flatten convolutional output
    flatten = layers.Flatten(name='flatten')(skip_3)

    # Secondary input: 15 metadata features
    input_2 = Input(shape=(15,), name='input_2')
    dense_input_2 = layers.Dense(15, activation='relu',
                                 kernel_regularizer=regularizers.l2(l2_reg), name='dense')(input_2)

    # Combine both inputs
    combined = layers.Concatenate(name='concatenate_3')([flatten, dense_input_2])

    # Final dense layers with dropout
    x = layers.Dense(4096, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg), name='dense_1')(combined)
    x = layers.Dropout(dropout_rate, name='dropout_final')(x)
    output = layers.Dense(1, name='output')(x)

    # Create model
    model = Model(
        inputs=[input_1, input_2],
        outputs=output,
        name='chess_cnn_regularized'
    )

    # Compile model
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model


def initialize_smaller_model(dropout_rate=0.6, l2_reg=0):
    """
    Initialize a SMALLER CNN model with fewer parameters to reduce overfitting.

    Key differences:
    - Only 2 residual blocks instead of 3
    - Smaller dense layer (2048 instead of 4096)
    - Dropout and L2 regularization included

    Args:
        dropout_rate (float): Dropout rate (0.0 to 1.0). Default 0.3
        l2_reg (float): L2 regularization strength. Default 0.001

    Returns:
        keras.Model: Compiled smaller CNN model
    """
    # Main input: chess board tensor (8x8x12)
    input_1 = Input(shape=(8, 8, 12), name='input_1')

    # First residual block
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(l2_reg), name='conv2d')(input_1)
    x = layers.BatchNormalization(name='batch_normalization')(x)
    x = layers.Dropout(dropout_rate, name='dropout_1')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(l2_reg), name='conv2d_1')(x)
    x = layers.BatchNormalization(name='batch_normalization_1')(x)
    x = layers.Dropout(dropout_rate, name='dropout_2')(x)
    skip_1 = layers.Concatenate(name='concatenate')([input_1, x])

    # Second residual block
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(l2_reg), name='conv2d_2')(skip_1)
    x = layers.BatchNormalization(name='batch_normalization_2')(x)
    x = layers.Dropout(dropout_rate, name='dropout_3')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(l2_reg), name='conv2d_3')(x)
    x = layers.BatchNormalization(name='batch_normalization_3')(x)
    x = layers.Dropout(dropout_rate, name='dropout_4')(x)
    skip_2 = layers.Concatenate(name='concatenate_1')([skip_1, x])

    # Flatten convolutional output
    flatten = layers.Flatten(name='flatten')(skip_2)

    # Secondary input: 15 metadata features
    input_2 = Input(shape=(15,), name='input_2')
    dense_input_2 = layers.Dense(15, activation='relu',
                                 kernel_regularizer=regularizers.l2(l2_reg), name='dense')(input_2)

    # Combine both inputs
    combined = layers.Concatenate(name='concatenate_2')([flatten, dense_input_2])

    # Smaller final dense layer
    x = layers.Dense(2048, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg), name='dense_1')(combined)
    x = layers.Dropout(dropout_rate, name='dropout_final')(x)
    output = layers.Dense(1, name='output')(x)

    # Create model
    model = Model(
        inputs=[input_1, input_2],
        outputs=output,
        name='chess_cnn_smaller'
    )

    # Compile model
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model


# ------------- Data Preparation (reuse from cnn_john_full) ----------------

def prepare_data(df, fen_column, target_column, metadata_columns=None, limit_data=True):
    """
    Prepare features (X) and target (y) from DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with chess positions and evaluations
        fen_column (str): Name of column containing FEN strings
        target_column (str): Name of column containing evaluation scores
        metadata_columns (list): List of metadata column names to include in X
        limit_data (bool): Whether to apply DATA_SIZE limit from params

    Returns:
        tuple: (X, y) - Features DataFrame and target Series
    """
    if limit_data:
        data_limit = get_data_size_limit()
        if data_limit is not None:
            df = df.head(data_limit)
            print(f"Data limited to {data_limit} rows (DATA_SIZE={DATA_SIZE})")

    if metadata_columns is not None:
        X = df[[fen_column] + metadata_columns]
    else:
        X = df[[fen_column]]

    y = df[target_column]

    return X, y


def create_train_test_split(X, y, test_size=None, random_state=None):
    """Split data into training and testing sets."""
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


def convert_fen_to_tensors(X_train, X_test, fen_column):
    """Convert FEN strings to tensor representations."""
    X_train_tensors = X_train[fen_column].apply(fen_to_tensor_8_8_12)
    X_test_tensors = X_test[fen_column].apply(fen_to_tensor_8_8_12)

    return X_train_tensors, X_test_tensors


def prepare_arrays_for_model(X_train_tensors, X_test_tensors, y_train, y_test):
    """Convert pandas Series to numpy arrays suitable for model input."""
    X_train_arr = np.stack(X_train_tensors.values).astype('float32')
    X_test_arr = np.stack(X_test_tensors.values).astype('float32')

    y_train_arr = y_train.values.astype('float32')
    y_test_arr = y_test.values.astype('float32')

    return X_train_arr, X_test_arr, y_train_arr, y_test_arr


def create_secondary_input(X_train, X_test, metadata_columns):
    """Create secondary input features for the model using REAL metadata from DataFrame."""
    X_train_input2 = X_train[metadata_columns].values.astype('float32')
    X_test_input2 = X_test[metadata_columns].values.astype('float32')

    return X_train_input2, X_test_input2


def clean_input_model(X, y, fen_column, metadata_columns):
    """Complete data preprocessing pipeline for model training with FULL metadata features."""
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    X_train_tensors, X_test_tensors = convert_fen_to_tensors(X_train, X_test, fen_column)
    X_train_arr, X_test_arr, y_train_arr, y_test_arr = prepare_arrays_for_model(
        X_train_tensors, X_test_tensors, y_train, y_test
    )
    X_train_input2, X_test_input2 = create_secondary_input(X_train, X_test, metadata_columns)

    X_train_inputs = [X_train_arr, X_train_input2]
    X_test_inputs = [X_test_arr, X_test_input2]

    return X_train_inputs, X_test_inputs, y_train_arr, y_test_arr


# ------------- Training with Enhanced Callbacks ----------------

def train_model_with_lr_schedule(model, X, y, fen_column, metadata_columns,
                                 validation_split=None, epochs=None, batch_size=None,
                                 patience=None, reduce_lr=True):
    """
    Train the CNN model with learning rate reduction on plateau.

    Enhanced training with:
    - Early stopping (restore best weights)
    - Learning rate reduction when validation loss plateaus

    Args:
        model (keras.Model): Initialized CNN model
        X (pd.DataFrame): Features (FEN strings + metadata columns)
        y (pd.Series): Target values (evaluation scores)
        fen_column (str): Name of FEN column in X
        metadata_columns (list): List of 15 metadata column names
        validation_split (float): Fraction of training data for validation
        epochs (int): Maximum number of training epochs
        batch_size (int): Number of samples per gradient update
        patience (int): Epochs to wait before early stopping
        reduce_lr (bool): Whether to use learning rate reduction

    Returns:
        tuple: (trained_model, history)
    """
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

    # Callbacks
    callbacks = []

    # Early stopping
    es = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(es)

    # Learning rate reduction on plateau
    if reduce_lr:
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_scheduler)

    # Train the model
    history = model.fit(
        X_train_inputs,
        y_train_arr,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


def evaluate_model(model, X, y, fen_column, metadata_columns):
    """Evaluate model performance on test data."""
    _, X_test_inputs, _, y_test_arr = clean_input_model(X, y, fen_column, metadata_columns)

    test_loss, test_mae = model.evaluate(X_test_inputs, y_test_arr, verbose=0)

    metrics = {
        'test_loss': test_loss,
        'test_mae': test_mae
    }

    print(f"\nTest Set Evaluation:")
    print(f"  Loss (MSE): {test_loss:.4f}")
    print(f"  MAE: {test_mae:.4f}")

    return metrics


# ------------- Analysis and Comparison Tools ----------------

def analyze_overfitting(history):
    """
    Analyze training history to quantify overfitting.

    Args:
        history: Keras History object from model.fit()

    Returns:
        dict: Overfitting metrics
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_mae = history.history['mae']
    val_mae = history.history['val_mae']

    # Find epoch with best validation loss
    best_epoch = np.argmin(val_loss)

    # Calculate gaps at best epoch
    loss_gap = val_loss[best_epoch] - train_loss[best_epoch]
    mae_gap = val_mae[best_epoch] - train_mae[best_epoch]

    # Calculate final gaps
    final_loss_gap = val_loss[-1] - train_loss[-1]
    final_mae_gap = val_mae[-1] - train_mae[-1]

    metrics = {
        'best_epoch': best_epoch + 1,
        'best_val_loss': val_loss[best_epoch],
        'best_train_loss': train_loss[best_epoch],
        'loss_gap_at_best': loss_gap,
        'mae_gap_at_best': mae_gap,
        'final_loss_gap': final_loss_gap,
        'final_mae_gap': final_mae_gap,
        'total_epochs': len(train_loss)
    }

    print("\n" + "=" * 60)
    print("OVERFITTING ANALYSIS")
    print("=" * 60)
    print(f"Best validation loss at epoch:  {metrics['best_epoch']}/{metrics['total_epochs']}")
    print(f"Train loss at best epoch:       {metrics['best_train_loss']:.4f}")
    print(f"Validation loss at best epoch:  {metrics['best_val_loss']:.4f}")
    print(f"Loss gap at best epoch:         {metrics['loss_gap_at_best']:.4f}")
    print(f"MAE gap at best epoch:          {metrics['mae_gap_at_best']:.4f}")
    print(f"\nFinal loss gap:                 {metrics['final_loss_gap']:.4f}")
    print(f"Final MAE gap:                  {metrics['final_mae_gap']:.4f}")

    if metrics['loss_gap_at_best'] > 1.0:
        print("\n⚠️  SEVERE OVERFITTING detected (gap > 1.0)")
    elif metrics['loss_gap_at_best'] > 0.5:
        print("\n⚠️  Moderate overfitting detected (gap > 0.5)")
    else:
        print("\n✓ Good generalization (gap < 0.5)")
    print("=" * 60)

    return metrics


def compare_models(histories, model_names):
    """
    Compare multiple model training histories.

    Args:
        histories (list): List of Keras History objects
        model_names (list): List of model names corresponding to histories

    Returns:
        pd.DataFrame: Comparison table
    """
    comparison = []

    for history, name in zip(histories, model_names):
        val_loss = history.history['val_loss']
        train_loss = history.history['loss']
        best_epoch = np.argmin(val_loss)

        comparison.append({
            'Model': name,
            'Best Epoch': best_epoch + 1,
            'Best Val Loss': val_loss[best_epoch],
            'Best Train Loss': train_loss[best_epoch],
            'Loss Gap': val_loss[best_epoch] - train_loss[best_epoch],
            'Total Epochs': len(val_loss)
        })

    df = pd.DataFrame(comparison)
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    return df


# ------------- Complete Workflows ----------------

def complete_workflow_regularized(df, fen_column, target_column, metadata_columns,
                                  model_type='dropout', dropout_rate=0.6, l2_reg=0):
    """
    Complete workflow with regularized model.

    Args:
        df (pd.DataFrame): Input DataFrame
        fen_column (str): Name of FEN column
        target_column (str): Name of evaluation column
        metadata_columns (list): List of 15 metadata column names
        model_type (str): 'dropout' or 'smaller'
        dropout_rate (float): Dropout rate for regularization
        l2_reg (float): L2 regularization strength

    Returns:
        tuple: (trained_model, history)
    """
    print("=" * 60)
    print(f"REGULARIZED MODEL TRAINING ({model_type.upper()})")
    print("=" * 60)

    # Prepare data
    print("\n[1/5] Preparing data...")
    X, y = prepare_data(df, fen_column, target_column, metadata_columns, limit_data=True)
    print(f"  Loaded {len(X)} positions")

    # Initialize model
    print(f"\n[2/5] Initializing {model_type} model...")
    if model_type == 'dropout':
        model = initialize_model_with_dropout(dropout_rate, l2_reg)
    elif model_type == 'smaller':
        model = initialize_smaller_model(dropout_rate, l2_reg)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"  Model created with {model.count_params():,} parameters")
    print(f"  Dropout rate: {dropout_rate}")
    print(f"  L2 regularization: {l2_reg}")

    # Train model
    print("\n[3/5] Training model...")
    trained_model, history = train_model_with_lr_schedule(
        model, X, y, fen_column, metadata_columns
    )
    print(f"  Training completed in {len(history.history['loss'])} epochs")

    # Analyze overfitting
    print("\n[4/5] Analyzing overfitting...")
    analyze_overfitting(history)

    # Evaluate model
    print("\n[5/5] Evaluating model...")
    evaluate_model(trained_model, X, y, fen_column, metadata_columns)

    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)

    return trained_model, history


def experiment_comparison(df, fen_column, target_column, metadata_columns):
    """
    Run experiments comparing different regularization approaches.

    Trains 3 models:
    1. Original (no regularization) - for comparison
    2. With dropout and L2 regularization
    3. Smaller architecture with dropout

    Args:
        df (pd.DataFrame): Input DataFrame
        fen_column (str): Name of FEN column
        target_column (str): Name of evaluation column
        metadata_columns (list): List of 15 metadata column names

    Returns:
        dict: Dictionary containing all models and histories
    """
    from beyond_the_board.models.cnn_john_full import (
        initialize_model_cnn_john_full,
        train_model
    )

    print("=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON EXPERIMENT")
    print("=" * 80)

    X, y = prepare_data(df, fen_column, target_column, metadata_columns, limit_data=True)

    results = {}

    # Model 1: Original (no regularization)
    print("\n\n### EXPERIMENT 1/3: ORIGINAL MODEL (NO REGULARIZATION) ###")
    model1 = initialize_model_cnn_john_full()
    model1, history1 = train_model(model1, X, y, fen_column, metadata_columns)
    results['original'] = {'model': model1, 'history': history1}

    # Model 2: With dropout and L2
    print("\n\n### EXPERIMENT 2/3: DROPOUT + L2 REGULARIZATION ###")
    model2 = initialize_model_with_dropout(dropout_rate=0.6, l2_reg=0)
    model2, history2 = train_model_with_lr_schedule(model2, X, y, fen_column, metadata_columns)
    results['dropout_l2'] = {'model': model2, 'history': history2}

    # Model 3: Smaller model
    print("\n\n### EXPERIMENT 3/3: SMALLER MODEL ###")
    model3 = initialize_smaller_model(dropout_rate=0.6, l2_reg=0)
    model3, history3 = train_model_with_lr_schedule(model3, X, y, fen_column, metadata_columns)
    results['smaller'] = {'model': model3, 'history': history3}

    # Compare results
    print("\n\n" + "=" * 80)
    compare_models(
        [history1, history2, history3],
        ['Original', 'Dropout+L2', 'Smaller']
    )

    return results
