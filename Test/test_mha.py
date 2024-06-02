import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Layers.multihead_attention as mha


def test_multihead_attention():
    mhas = mha.MultiHeadAttention(512, 8)
    q = tf.random.normal((1, 60, 512))
    k = tf.random.normal((1, 60, 512))
    v = tf.random.normal((1, 60, 512))
    out = mhas(q, k, v, mask=None)
    print(out.shape)


test_multihead_attention()
