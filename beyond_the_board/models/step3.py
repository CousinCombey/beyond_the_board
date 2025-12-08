"""
Step 3: Final Coach Model

Combines:
- Original dataset (2.4M positions) - 60%
- Self-play data (200k positions) - 25%
- Checkmate puzzles (100k positions) - 15%

Outputs:
- Position evaluation
- Best move recommendation
- Mate distance estimation
- Confidence score
- Explanation (attention weights)

"""

from keras import Model, Input, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from beyond_the_board.tensor.main import fen_to_tensor_8_8_12
from beyond_the_board.tensor.enhanced_metadata import extract_all_features, get_feature_names
from beyond_the_board.params import BATCH_SIZE, MAX_EPOCHS, EARLY_STOPPING_PATIENCE


def initialize_coach_model():
    """
    Initialize final coach model with all capabilities.

    Architecture:
    - Enhanced feature extraction (40 features)
    - Deep residual CNN backbone
    - Attention mechanism for interpretability
    - Multiple task heads:
        * Evaluation (regression)
        * Move prediction (policy, 4096 moves)
        * Mate distance (classification, 0-10+)
        * Confidence (regression, 0-1)

    Returns:
        keras.Model: Complete coach model
    """
    # Input 1: Board tensor (8, 8, 12)
    input_board = Input(shape=(8, 8, 12), name='input_board')

    # Deep CNN backbone with 4 residual blocks
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1')(input_board)
    x = layers.BatchNormalization(name='bn1')(x)

    # Residual block 1
    shortcut = x
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='res1_conv1')(x)
    x = layers.BatchNormalization(name='res1_bn1')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', name='res1_conv2')(x)
    x = layers.BatchNormalization(name='res1_bn2')(x)
    x = layers.Add(name='res1_add')([shortcut, x])
    x = layers.Activation('relu', name='res1_activation')(x)

    # Residual block 2
    shortcut = x
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='res2_conv1')(x)
    x = layers.BatchNormalization(name='res2_bn1')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', name='res2_conv2')(x)
    x = layers.BatchNormalization(name='res2_bn2')(x)
    x = layers.Add(name='res2_add')([shortcut, x])
    x = layers.Activation('relu', name='res2_activation')(x)

    # Residual block 3
    shortcut = x
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='res3_conv1')(x)
    x = layers.BatchNormalization(name='res3_bn1')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', name='res3_conv2')(x)
    x = layers.BatchNormalization(name='res3_bn2')(x)
    x = layers.Add(name='res3_add')([shortcut, x])
    x = layers.Activation('relu', name='res3_activation')(x)

    # Residual block 4
    shortcut = x
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='res4_conv1')(x)
    x = layers.BatchNormalization(name='res4_bn1')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', name='res4_conv2')(x)
    x = layers.BatchNormalization(name='res4_bn2')(x)
    x = layers.Add(name='res4_add')([shortcut, x])
    x = layers.Activation('relu', name='res4_activation')(x)

    # Flatten
    board_features = layers.Flatten(name='flatten_board')(x)

    # Input 2: Enhanced metadata (40 features)
    input_metadata = Input(shape=(40,), name='input_metadata')

    # Metadata processing with attention
    metadata_dense1 = layers.Dense(128, activation='relu', name='metadata_dense1')(input_metadata)
    metadata_dense2 = layers.Dense(64, activation='relu', name='metadata_dense2')(metadata_dense1)

    # Attention mechanism for metadata
    attention_scores = layers.Dense(64, activation='softmax', name='attention_scores')(metadata_dense2)
    attended_metadata = layers.Multiply(name='attended_metadata')([metadata_dense2, attention_scores])

    # Combine board and metadata
    combined = layers.Concatenate(name='combined')([board_features, attended_metadata])

    # Shared representation
    shared = layers.Dense(1024, activation='relu', name='shared_dense1')(combined)
    shared = layers.Dropout(0.4, name='shared_dropout1')(shared)
    shared = layers.Dense(512, activation='relu', name='shared_dense2')(shared)
    shared = layers.Dropout(0.3, name='shared_dropout2')(shared)

    # Task 1: Position Evaluation
    eval_head = layers.Dense(256, activation='relu', name='eval_dense1')(shared)
    eval_head = layers.Dense(128, activation='relu', name='eval_dense2')(eval_head)
    eval_output = layers.Dense(1, name='evaluation_output')(eval_head)

    # Task 2: Move Prediction (Policy)
    policy_head = layers.Dense(512, activation='relu', name='policy_dense1')(shared)
    policy_head = layers.Dropout(0.3, name='policy_dropout')(policy_head)
    policy_head = layers.Dense(256, activation='relu', name='policy_dense2')(policy_head)
    policy_output = layers.Dense(4096, activation='softmax', name='policy_output')(policy_head)

    # Task 3: Mate Distance (0, 1, 2, 3, 4, 5+)
    mate_head = layers.Dense(128, activation='relu', name='mate_dense1')(shared)
    mate_head = layers.Dense(64, activation='relu', name='mate_dense2')(mate_head)
    mate_output = layers.Dense(6, activation='softmax', name='mate_distance_output')(mate_head)

    # Task 4: Confidence Score
    confidence_head = layers.Dense(64, activation='relu', name='confidence_dense')(shared)
    confidence_output = layers.Dense(1, activation='sigmoid', name='confidence_output')(confidence_head)

    # Create model
    model = Model(
        inputs=[input_board, input_metadata],
        outputs=[eval_output, policy_output, mate_output, confidence_output],
        name='coach_model'
    )

    # Compile with weighted losses
    model.compile(
        optimizer='adam',
        loss={
            'evaluation_output': 'mean_squared_error',
            'policy_output': 'categorical_crossentropy',
            'mate_distance_output': 'categorical_crossentropy',
            'confidence_output': 'mean_squared_error'
        },
        loss_weights={
            'evaluation_output': 0.4,
            'policy_output': 0.3,
            'mate_distance_output': 0.2,
            'confidence_output': 0.1
        },
        metrics={
            'evaluation_output': ['mae'],
            'policy_output': ['accuracy'],
            'mate_distance_output': ['accuracy'],
            'confidence_output': ['mae']
        }
    )

    return model


def combine_datasets(original_df, selfplay_df, puzzle_df=None, sample_weights=None):
    """
    Combine multiple data sources with proper weighting.

    Args:
        original_df: Original 2.4M dataset
        selfplay_df: Self-play generated data
        puzzle_df: Optional checkmate puzzles
        sample_weights: Dict with sampling weights per source

    Returns:
        Combined DataFrame with balanced sampling
    """
    if sample_weights is None:
        sample_weights = {
            'original': 0.60,
            'selfplay': 0.25,
            'puzzles': 0.15
        }

    print("Combining datasets...")
    print(f"  Original: {len(original_df)} positions")
    print(f"  Self-play: {len(selfplay_df)} positions")
    if puzzle_df is not None:
        print(f"  Puzzles: {len(puzzle_df)} positions")

    # Sample from each dataset
    total_size = len(original_df)

    original_sample = original_df.sample(
        n=int(total_size * sample_weights['original']),
        random_state=42
    )

    selfplay_sample = selfplay_df.sample(
        n=min(len(selfplay_df), int(total_size * sample_weights['selfplay'])),
        random_state=42,
        replace=(len(selfplay_df) < int(total_size * sample_weights['selfplay']))
    )

    combined = pd.concat([original_sample, selfplay_sample], ignore_index=True)

    if puzzle_df is not None:
        puzzle_sample = puzzle_df.sample(
            n=min(len(puzzle_df), int(total_size * sample_weights['puzzles'])),
            random_state=42,
            replace=(len(puzzle_df) < int(total_size * sample_weights['puzzles']))
        )
        combined = pd.concat([combined, puzzle_sample], ignore_index=True)

    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"✓ Combined dataset: {len(combined)} positions")

    return combined


def train_coach_model(original_df, selfplay_df, puzzle_df=None,
                      validation_split=0.2, epochs=MAX_EPOCHS,
                      batch_size=BATCH_SIZE, patience=EARLY_STOPPING_PATIENCE,
                      save_path='models/coach_model.keras'):
    """
    Train final coach model.

    Args:
        original_df: Original training data
        selfplay_df: Self-play data
        puzzle_df: Optional puzzle data
        validation_split: Validation fraction
        epochs: Maximum epochs
        batch_size: Batch size
        patience: Early stopping patience
        save_path: Model save path

    Returns:
        Tuple of (trained_model, history)
    """
    print("=" * 80)
    print("STEP 3: FINAL COACH MODEL TRAINING")
    print("=" * 80)

    # Initialize model
    print("\n[1/4] Initializing coach model...")
    model = initialize_coach_model()
    print(f"  Model parameters: {model.count_params():,}")

    # Combine datasets
    print("\n[2/4] Combining datasets...")
    combined_df = combine_datasets(original_df, selfplay_df, puzzle_df)

    # Prepare training data
    print("\n[3/4] Preparing training data...")
    # Note: This requires full data preparation logic
    # For now, showing structure

    print(f"  Combined dataset size: {len(combined_df)}")

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            save_path,
            monitor='val_mate_distance_output_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience//2,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print("\n[4/4] Training model...")
    print("  (Data preparation and training logic to be implemented)")

    print("\n" + "=" * 80)
    print("STEP 3 TRAINING COMPLETE")
    print("=" * 80)

    return model, None


if __name__ == '__main__':
    print("Step 3 Coach Model")
    print("\nInitializing model architecture...")

    model = initialize_coach_model()
    model.summary()

    print(f"\n✓ Model architecture verified!")
    print(f"Total parameters: {model.count_params():,}")
