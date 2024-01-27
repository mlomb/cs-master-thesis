import chess
import tensorflow as tf
import numpy as np
from numpy import ndarray as NDArray

def decode_board(tensor: tf.Tensor) -> tf.Tensor:
    """
    Converts the 12 uint64s into a 768 float32 tensor.
    """
    masks = tf.convert_to_tensor(2 ** np.arange(64, dtype=np.int64))
    masked = tf.bitwise.bitwise_and(tf.expand_dims(tensor, -1), masks)
    expanded = tf.cast(tf.not_equal(masked, 0), dtype=tf.float32)
    return tf.reshape(expanded, (-1, 768))

# Expected empty board ecoding
# [71776119061217280, 4755801206503243776, 2594073385365405696, 9295429630892703744, 576460752303423488, 1152921504606846976, 65280, 66, 36, 129, 8, 16]
def encode_board(board: chess.Board):
    """
    Encode into a 12 uint64 tensor.
    """
    bytes = [0] * 8 * 12

    for i in range(64):
        piece = board.piece_at(i)
        if piece is not None:
            match piece.symbol():
                case "P": channel = 0
                case "N": channel = 1
                case "B": channel = 2
                case "R": channel = 3
                case "Q": channel = 4
                case "K": channel = 5
                case "p": channel = 6
                case "n": channel = 7
                case "b": channel = 8
                case "r": channel = 9
                case "q": channel = 10
                case "k": channel = 11

            rank = i // 8
            file = i % 8

            bytes[7 - rank + channel * 8] |= 1 << file

    return np.array(bytes, dtype=np.uint8).view(np.uint64)
