"""
Step 1: Multi-Task Chess Model with Enhanced Tactical Features

This model learns:
1. Position evaluation (regression)
2. Checkmate detection (binary classification)
3. Check detection (binary classification)
4. Game outcome prediction (3-class classification)

Architecture uses 40 metadata features including tactical awareness.

"""

from beyond_the_board.tensor.main import fen_to_tensor_8_8_12
from beyond_the_board.tensor.enhanced_metadata import extract_all_features, get_feature_names
from beyond_the_board.params import (
    RANDOM_SEED,
    TEST_SIZE,
    VALIDATION_SIZE,
    BATCH_SIZE,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE
)
from keras import Model, Input, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def initialize_step1_model():
    """
    Initialize multi-task model with 40 enhanced features.

    Architecture:
    - Input 1: (8, 8, 12) board tensor
    - Input 2: (40,) enhanced metadata features
    - Shared trunk: 3 residual CNN blocks
    - Task heads:
        * Evaluation head (regression)
        * Checkmate head (binary classification)
        * Check head (binary classification)
        * Outcome head (3-class classification: white/draw/black)

    Returns:
        keras.Model: Compiled multi-task model
    """
    # Input 1: Board tensor
    input_board = Input(shape=(8, 8, 12), name='input_board')

    # First residual block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_1')(input_board)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    skip_1 = layers.Concatenate(name='skip_1')([input_board, x])

    # Second residual block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_3')(skip_1)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_4')(x)
    x = layers.BatchNormalization(name='bn_4')(x)
    skip_2 = layers.Concatenate(name='skip_2')([skip_1, x])

    # Third residual block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_5')(skip_2)
    x = layers.BatchNormalization(name='bn_5')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_6')(x)
    x = layers.BatchNormalization(name='bn_6')(x)
    skip_3 = layers.Concatenate(name='skip_3')([skip_2, x])

    # Flatten CNN output
    flatten = layers.Flatten(name='flatten')(skip_3)

    # Input 2: Enhanced metadata (40 features)
    input_metadata = Input(shape=(40,), name='input_metadata')

    # Process metadata with attention mechanism
    metadata_dense = layers.Dense(64, activation='relu', name='metadata_dense_1')(input_metadata)
    metadata_dense = layers.Dropout(0.3, name='metadata_dropout')(metadata_dense)
    metadata_dense = layers.Dense(32, activation='relu', name='metadata_dense_2')(metadata_dense)

    # Combine board and metadata
    combined = layers.Concatenate(name='combined')([flatten, metadata_dense])

    # Shared dense layers
    shared = layers.Dense(512, activation='relu', name='shared_dense_1')(combined)
    shared = layers.Dropout(0.4, name='shared_dropout')(shared)
    shared = layers.Dense(256, activation='relu', name='shared_dense_2')(shared)

    # Task 1: Position Evaluation (regression)
    eval_head = layers.Dense(128, activation='relu', name='eval_dense')(shared)
    eval_output = layers.Dense(1, name='evaluation_output')(eval_head)

    # Task 2: Checkmate Detection (binary classification)
    checkmate_head = layers.Dense(64, activation='relu', name='checkmate_dense')(shared)
    checkmate_output = layers.Dense(1, activation='sigmoid', name='checkmate_output')(checkmate_head)

    # Task 3: Check Detection (binary classification)
    check_head = layers.Dense(64, activation='relu', name='check_dense')(shared)
    check_output = layers.Dense(1, activation='sigmoid', name='check_output')(check_head)

    # Task 4: Game Outcome Prediction (3-class classification)
    outcome_head = layers.Dense(64, activation='relu', name='outcome_dense')(shared)
    outcome_output = layers.Dense(3, activation='softmax', name='outcome_output')(outcome_head)

    # Create multi-task model
    model = Model(
        inputs=[input_board, input_metadata],
        outputs=[eval_output, checkmate_output, check_output, outcome_output],
        name='step1_multitask_chess_model'
    )

    # Compile with multiple losses and metrics
    model.compile(
        optimizer='adam',
        loss={
            'evaluation_output': 'mean_squared_error',
            'checkmate_output': 'binary_crossentropy',
            'check_output': 'binary_crossentropy',
            'outcome_output': 'categorical_crossentropy'
        },
        loss_weights={
            'evaluation_output': 0.4,  # 40% weight
            'checkmate_output': 0.2,   # 20% weight
            'check_output': 0.2,       # 20% weight
            'outcome_output': 0.2      # 20% weight
        },
        metrics={
            'evaluation_output': ['mae'],
            'checkmate_output': ['accuracy', 'precision', 'recall'],
            'check_output': ['accuracy'],
            'outcome_output': ['accuracy']
        }
    )

    return model


def prepare_step1_data(df, fen_column='FEN', eval_column='Stockfish', result_column='result'):
    """
    Prepare data for Step 1 training with all tasks.

    Args:
        df: DataFrame with FEN, evaluation, and game result
        fen_column: Name of FEN column
        eval_column: Name of evaluation column (Stockfish score)
        result_column: Name of result column (game outcome)

    Returns:
        Tuple of (X_inputs, y_outputs_dict)
    """
    print(f"Preparing Step 1 training data from {len(df)} positions...")

    # Extract enhanced features
    print("  Extracting enhanced metadata...")
    features_list = []
    board_tensors = []
    valid_indices = []

    for idx, fen in enumerate(df[fen_column]):
        if idx % 10000 == 0:
            print(f"    Processed {idx}/{len(df)}...")

        try:
            # Get board tensor
            tensor = fen_to_tensor_8_8_12(fen)
            board_tensors.append(tensor)

            # Get metadata features
            features = extract_all_features(fen)
            feature_names = get_feature_names()
            feature_vector = [features[name] for name in feature_names]
            features_list.append(feature_vector)

            valid_indices.append(idx)

        except Exception as e:
            print(f"    Error at index {idx}: {e}")
            continue

    # Convert to arrays
    X_board = np.array(board_tensors, dtype=np.float32)
    X_metadata = np.array(features_list, dtype=np.float32)

    # Filter dataframe to valid indices
    df_valid = df.iloc[valid_indices].reset_index(drop=True)

    # Prepare target labels
    print("  Preparing target labels...")

    # Task 1: Evaluation scores
    y_evaluation = df_valid[eval_column].values.astype(np.float32)

    # Task 2: Checkmate labels (from metadata)
    y_checkmate = X_metadata[:, get_feature_names().index('is_checkmate')]

    # Task 3: Check labels (from metadata)
    y_check = X_metadata[:, get_feature_names().index('is_in_check')]

    # Task 4: Game outcome (encode result column)
    outcome_map = {
        '1-0': [1, 0, 0],  # White wins
        '0-1': [0, 0, 1],  # Black wins
        '1/2-1/2': [0, 1, 0],  # Draw
        'draw': [0, 1, 0]
    }

    y_outcome = []
    for result in df_valid[result_column]:
        # Handle different result formats
        result_str = str(result).strip()
        if 'white' in result_str.lower() or result_str == '1-0':
            y_outcome.append([1, 0, 0])
        elif 'black' in result_str.lower() or result_str == '0-1':
            y_outcome.append([0, 0, 1])
        else:  # draw or unknown
            y_outcome.append([0, 1, 0])

    y_outcome = np.array(y_outcome, dtype=np.float32)

    print(f"✓ Prepared data: {len(X_board)} positions")
    print(f"  - Board tensors: {X_board.shape}")
    print(f"  - Metadata features: {X_metadata.shape}")
    print(f"  - Checkmate rate: {y_checkmate.mean()*100:.1f}%")
    print(f"  - Check rate: {y_check.mean()*100:.1f}%")
    print(f"  - Outcome distribution: W={y_outcome[:,0].mean():.2f} D={y_outcome[:,1].mean():.2f} B={y_outcome[:,2].mean():.2f}")

    X_inputs = [X_board, X_metadata]
    y_outputs = {
        'evaluation_output': y_evaluation,
        'checkmate_output': y_checkmate,
        'check_output': y_check,
        'outcome_output': y_outcome
    }

    return X_inputs, y_outputs


def train_step1_model(df, fen_column='FEN', eval_column='Stockfish', result_column='result',
                      test_size=TEST_SIZE, validation_split=VALIDATION_SIZE,
                      epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
                      patience=EARLY_STOPPING_PATIENCE, save_path='models/step1_model.keras'):
    """
    Complete training workflow for Step 1 model.

    Args:
        df: DataFrame with training data
        fen_column: Name of FEN column
        eval_column: Name of evaluation column
        result_column: Name of result column
        test_size: Fraction for test set
        validation_split: Fraction of training for validation
        epochs: Maximum epochs
        batch_size: Batch size
        patience: Early stopping patience
        save_path: Path to save best model

    Returns:
        Tuple of (trained_model, history, test_metrics)
    """
    print("=" * 80)
    print("STEP 1: MULTI-TASK MODEL TRAINING")
    print("=" * 80)

    # Initialize model
    print("\n[1/5] Initializing model...")
    model = initialize_step1_model()
    print(f"  Model created with {model.count_params():,} parameters")
    model.summary()

    # Prepare data
    print("\n[2/5] Preparing data...")
    X_inputs, y_outputs = prepare_step1_data(df, fen_column, eval_column, result_column)

    # Train-test split
    print("\n[3/5] Creating train-test split...")
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(X_inputs[0]))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=RANDOM_SEED)

    X_train = [X_inputs[0][train_idx], X_inputs[1][train_idx]]
    X_test = [X_inputs[0][test_idx], X_inputs[1][test_idx]]

    y_train = {k: v[train_idx] for k, v in y_outputs.items()}
    y_test = {k: v[test_idx] for k, v in y_outputs.items()}

    print(f"  Training samples: {len(train_idx)}")
    print(f"  Test samples: {len(test_idx)}")

    # Callbacks
    print("\n[4/5] Training model...")
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            save_path,
            monitor='val_checkmate_output_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train
    history = model.fit(
        X_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n[5/5] Evaluating on test set...")
    test_metrics = model.evaluate(X_test, y_test, verbose=1)

    print("\n" + "=" * 80)
    print("STEP 1 TRAINING COMPLETE")
    print("=" * 80)
    print(f"Model saved to: {save_path}")
    print(f"Test metrics: {test_metrics}")

    return model, history, test_metrics


if __name__ == '__main__':
    # Test with sample data
    print("Testing Step 1 model architecture...\n")

    # Create sample data
    sample_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "8/5k1p/7Q/7K/5Pq1/4n3/PPP5/8 w - - 2 48",  # Checkmate
        "rnbqkb1r/pppp1ppp/5n2/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 3"  # Check
    ]

    sample_df = pd.DataFrame({
        'FEN': sample_fens,
        'Stockfish': [0.0, 10.0, 1.5],
        'result': ['1/2-1/2', '1-0', '1/2-1/2']
    })

    # Initialize model
    model = initialize_step1_model()
    model.summary()

    print("\n✓ Model architecture verified!")
    print(f"Total parameters: {model.count_params():,}")
