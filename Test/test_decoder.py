import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Layers.decoder as dec

def test():
    d_model = 512
    num_heads = 8
    dff = 2048
    target_vocab_size = 8000
    maximum_position_encoding = 5000
    num_decoder_layers = 2
    rate = 0.1
    decoder = dec.DecoderPack(num_decoder_layers, d_model, num_heads, dff,
                          target_vocab_size, maximum_position_encoding, rate)
    target = tf.random.uniform((64, 60), dtype=tf.int64, minval=0, maxval=200)
    enc_output = tf.random.uniform((64, 60, 512))
    look_ahead_mask = tf.random.uniform((64, 1, 1, 60))
    padding_mask = tf.random.uniform((64, 1, 1, 60))
    out = decoder(target, enc_output, training=True,
                  look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)

    print(out.shape)

test()