import tensorflow as tf
import positioncal_encoding as pe
import multihead_attention as mha
from keras import Model


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = mha.MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def __call__(self, x, training, mask=None):
        inp1 = self.mha(x, x, x, mask=mask)
        out1 = self.dropout1(inp1, training=training)
        out1 = self.layernorm1(out1 + x)

        inp2 = self.ffn(out1)
        out2 = self.dropout2(inp2, training=training)
        out2 = self.layernorm2(out2 + out1)
        return out2


class TransformerEncoderPack(tf.keras.layers.Layer):
    def __init__(self, num_encoder_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(TransformerEncoderPack, self).__init__()
        # Lập trình tại đây
        self.num_encoder_layers = num_encoder_layers
        self.embedding = tf.keras.layers.Embedding(
            input_vocab_size, d_model, input_length=maximum_position_encoding)
        self.pos_encoding = pe.positional_encoding(
            maximum_position_encoding, d_model)
        self.enc_layers = EncoderLayer(d_model, num_heads, dff, rate)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.global_average_pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.final_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, x, training, mask=None):
        x = self.embedding(x)
        # print(tf.shape(x))
        x = x + self.pos_encoding[:, :tf.shape(x)[1], :]
        x = self.dropout(x, training=training)
        for _ in range(self.num_encoder_layers):
            x = self.enc_layers(x, training=training, mask=mask)

        return x
