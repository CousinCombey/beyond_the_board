# function that can transform a FEN string in a tensor object of shape (8, 8)

import chess
import torch
import numpy as np


def fen_to_tensor_8_8(fen):

    """Convert a FEN string to a tensor representation of the chess board.

    Args:
        fen (str): The FEN string representing the chess board.

    Returns:
        torch.Tensor: A tensor of shape (8, 8) representing the chess board.
                      Each piece is represented by an integer code.
    """

    piece_values = {
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,  # Black pieces (negative)
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,        # White pieces (positive)
    }

    pieces = fen.split(" ")[0]
    rows = pieces.split("/")

    tensor = np.zeros((8, 8), dtype=np.int8)

    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                tensor[i, col] = piece_values[char]
                col += 1

    return tensor

#------------------------------------------------------------------------------

def fen_to_tensor_8_8_12(fen):

    """Convert a FEN string to a tensor representation of the chess board.

    Args:
        fen (str): The FEN string representing the chess board.

    Returns:
        torch.Tensor: A tensor of shape (8, 8, 12) representing the chess board.
                      Each piece is represented by an integer code.
    """

    piece_to_channel = {
            'p': 0,  # Black pawn
            'n': 1,  # Black knight
            'b': 2,  # Black bishop
            'r': 3,  # Black rook
            'q': 4,  # Black queen
            'k': 5,  # Black king
            'P': 6,  # White pawn
            'N': 7,  # White knight
            'B': 8,  # White bishop
            'R': 9,  # White rook
            'Q': 10, # White queen
            'K': 11, # White king
        }

    # Get piece placement (first part of FEN)
    pieces = fen.split(" ")[0]
    rows = pieces.split("/")

    tensor = np.zeros((8, 8, 12), dtype=np.float32)

    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                # Empty squares - skip ahead
                col += int(char)
            else:
                # Place piece in corresponding channel
                channel = piece_to_channel[char]
                tensor[i, col, channel] = 1.0
                col += 1

    return tensor


if __name__ == '__main__':
    # Example usage
    fen = "r1bqkbnr/pppppppp/n7/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    tensor_8_8 = fen_to_tensor_8_8(fen)
    tensor_8_8_12 = fen_to_tensor_8_8_12(fen)
    print("Tensor (8, 8):")
    print(tensor_8_8)
    print("\nTensor (8, 8, 12):")
    print(tensor_8_8_12)
    print("Shapes:", tensor_8_8.shape, tensor_8_8_12.shape)
