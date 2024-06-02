import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Layers.encoder as enc

def test_encoder():
    d_model = 512
    num_heads = 8
    num_layers = 6
    dff = 2048
    input_vocab_size = 8500
    pe_input = 10000
    rate = 0.1
    encoder = enc.EncoderPack(num_layers, d_model, num_heads,
                              dff, input_vocab_size, pe_input, rate)
    input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)
    output = encoder(input, training=False, mask=None)
    print(output.shape)  # (batch_size, input_seq_len, d_model)
    return output


test_encoder()
