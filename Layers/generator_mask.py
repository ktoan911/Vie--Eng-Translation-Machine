import numpy as np
import tensorflow as tf


def generate_padding_mask(inp):
    # tạo 1 ma trận sao cho chiếu theo ma trận gốc, nếu giá trị tại vị trí đó là 0 thì giá trị tương ứng trong ma trận mới là 1, ngược lại là 0

    result = tf.cast(inp == 0, dtype=tf.float32)[:, np.newaxis, np.newaxis, :]
    return result


def generate_look_ahead_mask(inp_len):
    # tạo 1 mask có kích thước (inp_len, inp_len) với các giá trị ở phía dưới đường chéo chính là 0, trên đường chéo chính là 1
    mask = 1 - tf.linalg.band_part(tf.ones((inp_len, inp_len)), -1, 0)
    return mask


def generate_mask(inp, targ):
    # TODO: Update document
    # Encoder Padding Mask
    encoder_padding_mask = generate_padding_mask(inp)

    # Decoder Padding Mask: Use for global multi head attention for masking encoder output
    decoder_padding_mask = generate_padding_mask(inp)

    # Look Ahead Padding Mask
    decoder_look_ahead_mask = generate_look_ahead_mask(targ.shape[1])

    # Decoder Padding Mask
    decoder_inp_padding_mask = generate_padding_mask(targ)

    # Combine Look Ahead Padding Mask and Decoder Padding Mask
    decoder_look_ahead_mask = tf.maximum(
        decoder_look_ahead_mask, decoder_inp_padding_mask)

    return encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask
