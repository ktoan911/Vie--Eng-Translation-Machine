import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Layers.generator_mask as gm

# Sample input
inp = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0]], dtype=tf.int32)
targ = tf.constant([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]], dtype=tf.int32)
encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask = gm.generate_mask(
    inp, targ)
print(encoder_padding_mask)
print(decoder_look_ahead_mask)
print(decoder_padding_mask)